import torch

from mmdet.ops.nms import nms_wrapper


def multiclass_nms(multi_bboxes,
                   multi_scores,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None,
                   roi_feats = None, # 新加入的参数   为了得到预测框所对应的map
                   rois = None,
                   bbox_pred = None,
                   cls_score = None
                   ):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (Tensor): shape (n, #class*4) or (n, 4)
        multi_scores (Tensor): shape (n, #class), where the 0th column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms_thr (float): NMS IoU threshold
        max_num (int): if there are more than max_num bboxes after NMS,
            only top max_num will be kept.
        score_factors (Tensor): The factors multiplied to scores before
            applying NMS

    Returns:
        tuple: (bboxes, labels), tensors of shape (k, 5) and (k, 1). Labels
            are 0-based.
    """
    # 排除背景类之后的剩余类数量
    # size(int) 沿着某个轴计算size
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    #
    if multi_bboxes.shape[1] > 4:
        # 前4个列算作背景类去掉,后边的是物体
        # shape: [multi_bboxes数量,物体类数*4]
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)[:, 1:]
    else:
        bboxes = multi_bboxes[:, None].expand(-1, num_classes, 4)

    #去除第一列的背景分数, 保留的列是各类物体的分数
    # shape: [multi_bboxes数量,物体类数]
    scores = multi_scores[:, 1:]

    # filter out boxes with low scores
    ''' 
        # 低分数过滤
        # 过滤掉分数 < score_thr的行
        valid_mask打印：
        tensor([[ True],
                [ True],
                [ True],
                [False],
        shape: [multi_bboxes数量,1]
    '''
    valid_mask = scores > score_thr
    # bboxes对应scores保留行
    # shape: [过滤后保留行数,物体类数*4]
    bboxes = bboxes[valid_mask]


    if score_factors is not None:
        # 默认为空 不会执行 fcos_head.py的时候会用上
        scores = scores * score_factors[:, None]
    # 过滤掉小于score_thr的行
    scores = scores[valid_mask]
    # 保留非零行
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        # bboxes数为0时 才会执行这里
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, ), dtype=torch.long)
        return bboxes, labels

    # Modified from https://github.com/pytorch/vision/blob
    # /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap

    # 同一类物体的框重叠过多就会被NMS抑制
    # print()
    # print("------------------------------------bbox_nms.py  1111---------------------------------")
    # print("===multi_bboxes:", multi_bboxes.shape)
    # print("===multi_scores:", multi_scores.shape)
    # print("====roi_feats[0]:",roi_feats[0].shape)
    # print("===roi_feats:",roi_feats.shape)
    # print("===filter_low_score_roi_feats",filter_low_score_roi_feats.shape)
    # print()
    # print("===valid_mask:", valid_mask.shape)
    # print("===bboxes:", bboxes.shape)
    # print("===labels:", labels.shape)
    # print("--------------------------------------------------------------------------------------")
    # print()

    max_coordinate = bboxes.max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    bboxes_for_nms = bboxes + offsets[:, None]
    nms_cfg_ = nms_cfg.copy()
    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = getattr(nms_wrapper, nms_type)
    # nms_op：NMS操作(具体注释,输入,输出格式 进入上边一行的nms_wrapper里看)
    # dets是NMS抑制后留下的bbox, keep是保留的行索引
    # dets是转换为科学计数法之后的box矩阵
    dets, keep = nms_op(
        torch.cat([bboxes_for_nms, scores[:, None]], 1),  # 这个cat操作将bbox和score按列拼接到一起 (?,4) + (?,1) ---> (?,5)
        **nms_cfg_)
    bboxes = bboxes[keep]
    scores = dets[:, -1]  # soft_nms will modify scores
    labels = labels[keep]


    top_max_inds = None
    if keep.size(0) > max_num:
        # 保存前 max_num个框
        _, inds = scores.sort(descending=True)
        inds = inds[:max_num]
        bboxes = bboxes[inds]
        scores = scores[inds]
        labels = labels[inds]
        top_max_inds = inds

    if roi_feats is not None:
        # 自定义函数
        # 保存最后识别的框 和 特征
        get_final_area(valid_mask,
                       roi_feats, # 新加入的参数   为了得到预测框所对应的map
                       rois,
                       bbox_pred,
                       cls_score,
                       top_max_inds)


    #
    # print()
    # print("------------------------------------bbox_nms.py  2222---------------------------------")
    # print("===max_coordinate:", max_coordinate)
    # print("===offsets:", offsets)
    # print("===bboxes_for_nms:", bboxes_for_nms)
    # print("===nms_cfg_:", nms_cfg_ )
    # print("===nms_type:", nms_type)
    # print("===nms_op:", nms_op)
    # print("--------")
    # print("===dets:", dets.shape, dets)
    # print("===keep(NMS的 inds):", keep.shape, keep)
    # print("--------")
    # print("===scores:", scores.shape, scores)
    # print("===labels:",labels.shape,labels)
    # print("===bboxes:", bboxes.shape,bboxes)
    # print("===final_roi_feats",final_roi_feats.shape)
    # print("===final_rois:",final_rois.shape)
    # print("--------------------------------------------------------------------------------------")
    # print()

    return torch.cat([bboxes, scores[:, None]], 1), labels

def get_final_area(valid_mask = None,  # 过滤掉低分
                   roi_feats = None,   # 新加入的参数   为了得到预测框所对应的map
                   rois = None,
                   bbox_pred = None,
                   cls_score = None,
                   top_max_inds = None # 根据参数保留前top_max个框
                   ):

    # 自定义函数
    # 保存最后识别的框 和 特征

    '''
    *********此处自己加上的
    # 过滤低分数的框后保留对应的roi_feats、roi
    '''
    i = 0
    idns = [] # 保留此索引所对应的行
    for one_row in valid_mask:
        if(one_row[0] == True):
            idns.append(i)
            i = i + 1
    filter_low_score_roi_feats = roi_feats[idns]
    filter_low_score_rois = rois[idns]
    filter_low_score_bbox_pred = bbox_pred[idns]
    filter_low_score_cls_score = cls_score[idns]
    '''
    *********
    '''

    # 为了创建引用
    final_roi_feats = filter_low_score_roi_feats
    final_rois = filter_low_score_rois
    final_bbox_pred = filter_low_score_bbox_pred
    final_cls_score = filter_low_score_cls_score

    if top_max_inds is not None:
        final_roi_feats = filter_low_score_roi_feats[top_max_inds]
        final_rois = filter_low_score_rois[top_max_inds]
        final_bbox_pred = filter_low_score_bbox_pred[top_max_inds]
        final_cls_score = filter_low_score_cls_score[top_max_inds]


    # 保存张量
    root_path = "/content/mmdetection/"
    picture_name = "Z108"
    save_path = root_path + picture_name + "_filter_final_roi_feats.pt"
    torch.save(final_roi_feats,save_path)
    save_path = root_path + picture_name + "_filter_final_rois.pt"
    torch.save(final_rois,save_path)
    save_path = root_path + picture_name + "_filter_final_bbox_pred.pt"
    torch.save(final_bbox_pred,save_path)
    save_path = root_path + picture_name + "_filter_final_cls_score.pt"
    torch.save(final_cls_score,save_path)