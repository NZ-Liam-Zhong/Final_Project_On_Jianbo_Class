from dataset import *
from backbone import *
from model import SOLO
from visualize import visualize_inference


if __name__ == "__main__":
    torch.random.manual_seed(1)

    # Prepare the dataset
    dataset = BuildDataset("hw3 solo datasets")

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    batch_size = 2
    NUM_VISUALIZED_IMAGES = 6
    train_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = BuildDataLoader(test_dataset, batch_size=NUM_VISUALIZED_IMAGES, shuffle=False, num_workers=0)

    _, visualization_sample = next(enumerate(test_loader))

    # Load the checkpoint
    backbone = Resnet50Backbone()
    model: SOLO = SOLO.load_from_checkpoint("checkpoints/solo_epoch=39.ckpt", strict=False).cuda()

    # Visualize
    img, label_list, mask_list, bbox_list = visualization_sample
    with torch.no_grad():
        cate_pred_list, ins_pred_list = model.forward(img.cuda())
    nms_sorted_scores, nms_sorted_labels, nms_sorted_ins = model.head.process_predictions(ins_pred_list, cate_pred_list, (img.shape[-2], img.shape[-1]))

    visualize_inference(img.cuda(), nms_sorted_labels, nms_sorted_ins)




