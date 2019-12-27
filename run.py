import numpy as np
import json
import argparse
import operator
from CIFAR_DataLoader import CifarDataManager, display_cifar
from vggNet import Model, model_structure
from multiprocessing import Pool

# def calculate_total_by_weights(class_id, count):
#     name_shape_match = [
#         {
#             "name": layer["name"],
#             "shape": [0, ] * layer["shape"],
#         }
#         for layer in model_structure
#     ]
#     for pic_id in range(count):
#         jsonpath = "./ImageEncoding/class" + str(class_id) + "-pic" + str(pic_id) + ".json"
#         with open(jsonpath, "r") as f:
#             dataset = json.load(f)
#             assert len(dataset) == len(name_shape_match)
#             for layer_id in range(len(dataset)):
#                 assert name_shape_match[layer_id]["name"] == dataset[layer_id]["layer_name"]
#                 name_shape_match[layer_id]["shape"] = map(
#                     operator.add,
#                     name_shape_match[layer_id]["shape"],
#                     dataset[layer_id]["layer_lambda"]
                # )
    # generator -> list, for json dump
    # for i in range(len(name_shape_match)):
    #     name_shape_match[i]["shape"] = list(name_shape_match[i]["shape"])
    # json_write_path = "./ClassEncoding/class" + str(class_id) + ".json"
    # with open(json_write_path, "w") as g:
    #     json.dump(name_shape_match, g, sort_keys=True, indent=4, separators=(",", ":"))


def parse_args():
    parser = argparse.ArgumentParser(description="Critical Path Pruning")
    parser.add_argument("classes", metavar="C", type=int, nargs="+",
                        help="Image class in CIFAR-100 (-1 for using all classes)")
    parser.add_argument("-r", "--learning_rate", help="Learning rate in optimizing control gates",
                        default=0.1, required=False)
    parser.add_argument("-t", "--threshold", help="Threshold for pruning",
                        default=0.0, required=False)
    parser.add_argument("-p", "--l1_loss_penalty", help="L1-loss prnalty in optimizing control gates",
                        default=0.03, required=False)
    parser.add_argument("-n", "--number", help="Number of images in each class", default=500, type=int, required=False)
    return parser.parse_args()


def process(class_id):
    print("class", class_id)
    d = CifarDataManager()
    
    model = Model(
        learning_rate=args.learning_rate,
        L1_loss_penalty=args.l1_loss_penalty,
        threshold=args.threshold
    )
    # print(class_id, args.number, args.learning_rate, args.l1_loss_penalty, args.threshold)
    
    train_images, train_labels = d.train.generateSpecializedData(class_id, args.number)
    model.encode_class_data(class_id, train_images)
    
    


if __name__ == "__main__":
    args = parse_args()
    target_classes = range(100) if args.classes[0] == -1 else args.classes
    
    po = Pool(4)
    for class_id in target_classes:
    	po.apply_async(process, (class_id,))
        # calculate_total_by_weights(class_id, args.number)
    po.close()
    po.join()
