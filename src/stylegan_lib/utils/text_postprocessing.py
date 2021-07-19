import numpy as np

def postprocess_text_logits(sent_pred, axes_range):
    # perform post-processing on text output logits
    # in order to align indices and scale logits
    # initialize processed logits array with -1
    proc_pred = np.array([-1.0]*32)
    # [0] bushy eyebrows attribute
    proc_pred[0] = sent_pred[0]
    # [1,2,3,4] hair color attributes
    if sent_pred[1] != -1:
        # black hair
        # NOTE : navigate in negative brown direction
        proc_pred[3] = 0
    else:
        # set hair color with output logits
        proc_pred[1] = sent_pred[2]
        proc_pred[2] = sent_pred[3]
        proc_pred[3] = sent_pred[4]
        proc_pred[4] = sent_pred[5]
    # [5] straight/curly hair attribute
    proc_pred[5] = sent_pred[6]
    # [6] receding hairline attribute
    proc_pred[6] = sent_pred[7]
    # [7] bald attribute
    proc_pred[7] = sent_pred[8]
    # [8] hair with bangs attribute
    proc_pred[8] = sent_pred[9]
    # [9] hair length attribute
    proc_pred[9] = sent_pred[10]
    # [10] beard attribute
    proc_pred[10] = sent_pred[11]
    # [11] skin color attribute
    proc_pred[11] = sent_pred[12]
    # [12] asian effect attribute
    proc_pred[12] = sent_pred[13]
    # [13] face thickness attribute
    proc_pred[13] = sent_pred[14]
    # [14] gender attribute
    proc_pred[14] = sent_pred[15]
    # [15] age attribute
    proc_pred[15] = sent_pred[16]
    # [16] eye bags attribute
    proc_pred[16] = sent_pred[17]
    # [17] eye width attribute
    proc_pred[17] = sent_pred[18]
    # [18,19,20] eye color attributes
    if sent_pred[19] != -1:
        # black eyes
        # NOTE : navigate in negative brown direction
        proc_pred[20] = 0
    else:
        # set eyes color with output logits
        proc_pred[18] = sent_pred[20]
        proc_pred[19] = sent_pred[21]
        proc_pred[20] = sent_pred[22]
    # [21] lip size attribute
    proc_pred[21] = sent_pred[23]
    # [22] nose size attribute
    proc_pred[22] = sent_pred[24]
    # [23] ear size attribute
    proc_pred[23] = sent_pred[25]
    # [24] double chin attribute
    proc_pred[24] = sent_pred[26]
    # [25] cheekbones attribute
    proc_pred[25] = sent_pred[27]
    # [26] nose tip attribute
    proc_pred[26] = sent_pred[28]
    # [27] rosy cheeks attribute
    proc_pred[27] = sent_pred[29]
    # [28] makeup attribute
    proc_pred[28] = sent_pred[30]
    # [29] lipstick attribute
    proc_pred[29] = sent_pred[31]
    # [30] sight glasses attribute
    proc_pred[30] = sent_pred[32]
    # [31] sun glasses attribute
    proc_pred[31] = sent_pred[33]
    # re-scale all attributes based on considered axes range
    proc_pred_scaled = np.array(
        [(logit*axes_range*2.0)-axes_range if logit != -1.0 else -100.0 for logit in proc_pred]
    )
    # return scaled processed text logits
    return proc_pred_scaled
