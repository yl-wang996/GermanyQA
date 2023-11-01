def calculate_em_f1(pred_starts,pred_ends,gt_starts,gt_ends):
    size = len(pred_starts)
    precisions = []
    recalls = []
    for idx in range(size):
        pred_start = pred_starts[idx]
        pred_end = pred_ends[idx]
        gt_start = gt_starts[idx]
        gt_end = gt_ends[idx]
        if pred_start > pred_end:
            pred_start,pred_end = pred_end,pred_start
        if gt_start > gt_end:
            gt_start,gt_end = gt_end,gt_start
        pred =set ([n for n in range(pred_start,pred_end+1,1)])
        gt = set([n for n in range(gt_start,gt_end+1,1)])
        cross = len(pred & gt)
        precision = cross/len(pred)
        recall = cross/len(gt)
        precisions.append(precision)
        recalls.append(recall)

    # calculate exact match and f1 score
    em_sum = 0
    f1_sum = 0
    for idx in range(size):
        if precisions[idx] == 1.0 and recalls[idx] == 1.0:
            em_sum += 1
        if precisions[idx]+recalls[idx] != 0:
            f1 = (2*precisions[idx]*recalls[idx])/(precisions[idx]+recalls[idx])
            f1_sum += f1
        else:
            f1_sum += 0

    return em_sum/size,f1_sum/size,sum(precisions)/size,sum(recalls)/size











