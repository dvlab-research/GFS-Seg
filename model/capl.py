import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import model.resnet as models
import random

manual_seed=321
torch.manual_seed(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.cuda.manual_seed_all(manual_seed)
random.seed(manual_seed)

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins, BatchNorm):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                BatchNorm(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)    


class PSPNet(nn.Module):
    def __init__(self, layers=50, bins=(1, 2, 3, 6), dropout=0.1, classes=2, \
        zoom_factor=8, criterion=nn.CrossEntropyLoss(ignore_index=255), \
        BatchNorm=nn.BatchNorm2d, pretrained=True, args=None):
        super(PSPNet, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.classes = classes
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins, BatchNorm)
        fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, 512, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, 256, kernel_size=1)
            )

        main_dim = 512
        aux_dim = 256
        self.main_proto = nn.Parameter(torch.randn(self.classes, main_dim).cuda())
        self.aux_proto = nn.Parameter(torch.randn(self.classes, aux_dim).cuda())
        gamma_dim = 1
        self.gamma_conv = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(512, gamma_dim)
        )

        self.args = args

    def forward(self, x, y=None, gened_proto=None, base_num=16, novel_num=5, iter=None, \
                gen_proto=False, eval_model=False, visualize=False):

        self.iter = iter
        self.base_num = base_num

        def WG(x, y, proto, target_cls):
            b, c, h, w = x.size()[:]
            tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h, w), mode='nearest') 
            out = x.clone()
            unique_y = list(tmp_y.unique())         
            new_gen_proto = proto.data.clone()
            for tmp_cls in unique_y:
                if tmp_cls == 255: 
                    continue
                tmp_mask = (tmp_y.float() == tmp_cls.float()).float()
                tmp_p = (out * tmp_mask).sum(0).sum(-1).sum(-1) / tmp_mask.sum(0).sum(-1).sum(-1)
                new_gen_proto[tmp_cls.long(), :] = tmp_p 
            return new_gen_proto        

        def generate_fake_proto(proto, x, y):
            b, c, h, w = x.size()[:]
            tmp_y = F.interpolate(y.float().unsqueeze(1), size=(h,w), mode='nearest')
            unique_y = list(tmp_y.unique())
            raw_unique_y = list(tmp_y.unique())
            if 0 in unique_y:
                unique_y.remove(0)
            if 255 in unique_y:
                unique_y.remove(255)

            novel_num = len(unique_y) // 2
            fake_novel = random.sample(unique_y, novel_num)
            for fn in fake_novel:
                unique_y.remove(fn) 
            fake_context = unique_y
            
            new_proto = self.main_proto.clone()
            new_proto = new_proto / (torch.norm(new_proto, 2, 1, True) + 1e-12)
            x = x / (torch.norm(x, 2, 1, True) + 1e-12)
            for fn in fake_novel:
                tmp_mask = (tmp_y == fn).float()
                tmp_feat = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)
                fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
                fake_vec[fn.long()] = 1
                new_proto = new_proto * (1 - fake_vec) + tmp_feat.unsqueeze(0) * fake_vec
            replace_proto = new_proto.clone()

            for fc in fake_context:
                tmp_mask = (tmp_y == fc).float()
                tmp_feat = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)              
                fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
                fake_vec[fc.long()] = 1
                raw_feat = new_proto[fc.long()].clone()
                all_feat = torch.cat([raw_feat, tmp_feat], 0).unsqueeze(0)  # 1, 1024
                ratio = F.sigmoid(self.gamma_conv(all_feat))[0]   # n, 512
                new_proto = new_proto * (1 - fake_vec) + ((raw_feat* ratio + tmp_feat* (1 - ratio)).unsqueeze(0) * fake_vec)

            if random.random() > 0.5 and 0 in raw_unique_y:
                tmp_mask = (tmp_y == 0).float()
                tmp_feat = (x * tmp_mask).sum(0).sum(-1).sum(-1) / (tmp_mask.sum(0).sum(-1).sum(-1) + 1e-12)  #512             
                fake_vec = torch.zeros(new_proto.size(0), 1).cuda()
                fake_vec[0] = 1
                raw_feat = new_proto[0].clone()
                all_feat = torch.cat([raw_feat, tmp_feat], 0).unsqueeze(0)  # 1, 1024         
                ratio = F.sigmoid(self.gamma_conv(all_feat))[0]   # 512
                new_proto = new_proto * (1 - fake_vec) + ((raw_feat * ratio + tmp_feat * (1 - ratio)).unsqueeze(0)  * fake_vec)

            return new_proto, replace_proto

        if gen_proto:
            # proto generation
            # supp_x: [cls, s, c, h, w]
            # supp_y: [cls, s, h, w]
            x = x[0]
            y = y[0]            
            cls_num = x.size(0)
            shot_num = x.size(1)
            with torch.no_grad():
                gened_proto = self.main_proto.clone()
                base_proto_list = []
                tmp_x_feat_list = []
                tmp_gened_proto_list = []
                for idx in range(cls_num):
                    tmp_x = x[idx] 
                    tmp_y = y[idx]
                    raw_tmp_y = tmp_y

                    tmp_x = self.layer0(tmp_x)
                    tmp_x = self.layer1(tmp_x)
                    tmp_x = self.layer2(tmp_x)
                    tmp_x = self.layer3(tmp_x)
                    tmp_x = self.layer4(tmp_x)
                    layer4_x = tmp_x
                    tmp_x = self.ppm(tmp_x)
                    ppm_feat = tmp_x.clone()
                    tmp_x = self.cls(tmp_x) 
                    tmp_x_feat_list.append(tmp_x)

                    tmp_cls = idx + base_num
                    tmp_gened_proto = WG(x=tmp_x, y=tmp_y, proto=self.main_proto, target_cls=tmp_cls)
                    tmp_gened_proto_list.append(tmp_gened_proto)
                    base_proto_list.append(tmp_gened_proto[:base_num, :].unsqueeze(0)) 
                    gened_proto[tmp_cls, :] = tmp_gened_proto[tmp_cls, :]


                base_proto = torch.cat(base_proto_list, 0).mean(0)  
                base_proto = base_proto / (torch.norm(base_proto, 2, 1, True) + 1e-12)
                ori_proto = self.main_proto[:base_num, :] / (torch.norm(self.main_proto[:base_num, :], 2, 1, True) + 1e-12)

                all_proto = torch.cat([ori_proto, base_proto], 1)
                ratio = F.sigmoid(self.gamma_conv(all_proto))   # n, 512
                base_proto = ratio * ori_proto + (1 - ratio) * base_proto
                gened_proto = torch.cat([base_proto, gened_proto[base_num:, :]], 0)           
                gened_proto = gened_proto / (torch.norm(gened_proto, 2, 1, True) + 1e-12)

            return gened_proto.unsqueeze(0)      


        else:
            x_size = x.size()
            assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
            h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
            w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)            
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x_tmp = self.layer3(x)
            x = self.layer4(x_tmp)
            x = self.ppm(x)
            x = self.cls(x)
            raw_x = x.clone()              


            if eval_model: 
                #### evaluation
                if len(gened_proto.size()[:]) == 3:
                    gened_proto = gened_proto[0]   
                if visualize:
                    vis_feat = x.clone()    

                refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=raw_x)
                refine_proto[:, :base_num] = refine_proto[:, :base_num] + gened_proto[:base_num].unsqueeze(0)
                refine_proto[:, base_num:] = refine_proto[:, base_num:] * 0 + gened_proto[base_num:].unsqueeze(0)
                x = self.get_pred(raw_x, refine_proto)
    
            else:
                ##### training
                fake_num = x.size(0) // 2              
                ori_new_proto, replace_proto = generate_fake_proto(proto=self.main_proto, x=x[fake_num:], y=y[fake_num:])                    
                x = self.get_pred(x, ori_new_proto)    

                x_pre = x.clone()
                refine_proto = self.post_refine_proto_v2(proto=self.main_proto, x=raw_x)
                post_refine_proto = refine_proto.clone()
                post_refine_proto[:, :base_num] = post_refine_proto[:, :base_num] + ori_new_proto[:base_num].unsqueeze(0)
                post_refine_proto[:, base_num:] = post_refine_proto[:, base_num:] * 0 + ori_new_proto[base_num:].unsqueeze(0)
                x = self.get_pred(raw_x, post_refine_proto)                                           


            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            if self.training:
                aux = self.aux(x_tmp)
                aux = self.get_pred(aux, self.aux_proto)
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                main_loss = self.criterion(x, y)
                aux_loss = self.criterion(aux, y)     
                          
                x_pre = F.interpolate(x_pre, size=(h, w), mode='bilinear', align_corners=True)
                pre_loss = self.criterion(x_pre, y)
                main_loss = 0.5 * main_loss + 0.5 * pre_loss
 
                return x.max(1)[1], main_loss, aux_loss
            else:
                if visualize:
                    return x, vis_feat
                else:
                    return x

    def post_refine_proto_v2(self, proto, x):
        raw_x = x.clone()        
        b, c, h, w = raw_x.shape[:]
        pred = self.get_pred(x, proto).view(b, proto.shape[0], h*w)
        pred = F.softmax(pred, 2)

        pred_proto = pred @ raw_x.view(b, c, h*w).permute(0, 2, 1)
        pred_proto_norm = F.normalize(pred_proto, 2, -1)    # b, n, c
        proto_norm = F.normalize(proto, 2, -1).unsqueeze(0)  # 1, n, c
        pred_weight = (pred_proto_norm * proto_norm).sum(-1).unsqueeze(-1)   # b, n, 1
        pred_weight = pred_weight * (pred_weight > 0).float()
        pred_proto = pred_weight * pred_proto + (1 - pred_weight) * proto.unsqueeze(0)  # b, n, c
        return pred_proto

    def get_pred(self, x, proto):
        b, c, h, w = x.size()[:]
        if len(proto.shape[:]) == 3:
            # x: [b, c, h, w]
            # proto: [b, cls, c]  
            cls_num = proto.size(1)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, -1, True)  # b, n, c
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            pred = proto @ x  # b, cls, hw
        elif len(proto.shape[:]) == 2:         
            cls_num = proto.size(0)
            x = x / torch.norm(x, 2, 1, True)
            proto = proto / torch.norm(proto, 2, 1, True)
            x = x.contiguous().view(b, c, h*w)  # b, c, hw
            proto = proto.unsqueeze(0)  # 1, cls, c
            pred = proto @ x  # b, cls, hw
        pred = pred.contiguous().view(b, cls_num, h, w)
        return pred * 10

