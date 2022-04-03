# https://github.com/ZhugeKongan/torch-template-for-deep-learning/blob/main/visualization.py
def draw_features(width, height, channels,x,savename):
    fig = plt.figure(figsize=(32,32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(channels):
        plt.subplot(height,width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = (img - pmin) / (pmax - pmin + 0.000001)
        plt.imshow(img, cmap='gray')
#         print("{}/{}".format(i, channels))
    fig.savefig(savename, dpi=300)
    fig.clf()
    plt.close()
    

def visualize_feature_map(img_batch):
    """Constructs a ECA module.
            Args:
                input: feature[B,H,W,C],img_size
               output: NONE
            """
    feature_map = img_batch[0].detach().cpu().numpy()
    print(feature_map.shape)

    feature_map_combination = []
    plt.figure()

    num_pic = feature_map.shape[2] #C
    row, col = get_row_col(num_pic) #图片数

    for i in range(0, num_pic):
        feature_map_split = feature_map[:, :, i]
        feature_map_combination.append(feature_map_split)
        plt.subplot(row, col, i + 1)
        plt.imshow(feature_map_split)
        plt.axis('off')
        plt.title('feature_map_{}'.format(i))

    plt.savefig('feature_map.png')
    plt.show()

    # 各个特征图按1：1 叠加
    feature_map_sum = sum(ele for ele in feature_map_combination)
    plt.imshow(feature_map_sum)
    plt.savefig("feature_map_sum.png")
