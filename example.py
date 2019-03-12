import fgs
import imagenet_labels
from PIL import Image
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models
import torch
import torch.nn.functional as F
import util


CUDA = True


def classify(model, im):
    """
    Classify the image with model and return the label and class
    probability
    """
    device = torch.device("cuda" if CUDA else "cpu")
    model.to(device)
    im = im.to(device)
    model.eval()
    out = model(im)
    label = imagenet_labels.label(out.argmax().item())
    prob = F.softmax(out, dim=1)
    return label, round((torch.max(prob.data, 1)[0].item()) * 100, 2)


# Load pretrained model
model = models.resnet34(pretrained=True)

# Load and classify image
im = Image.open("cat.jpg")  # taken from ImageNet test set
im = im.resize((224, 224), Image.ANTIALIAS)
true_label, true_prob = classify(model, util.preprocess(im))
print("True label: {}, prob: {}".format(true_label, true_prob))

# Generate adversarial example that will correspond to target_class
target_class = 800 #5
print("Target class:", imagenet_labels.label(target_class))
adverserial_image = fgs.fgs(model, util.preprocess(im), target_class,
                            targeted=True, alpha=0.01, iterations=10,
                            use_cuda=CUDA)
adverserial_image = util.postprocess(adverserial_image)
adv_label, adv_prob = classify(model, util.preprocess(adverserial_image))
print("Predicted label: {}, prob: {}".format(adv_label, adv_prob))

# Plot results
plt.subplot(131)
plt.title("Before: {} {}%".format(true_label, true_prob))
before_im = np.array(im)
plt.imshow(before_im)
plt.subplot(132)
plt.title("After: {} {}%".format(adv_label, adv_prob))
after_im = np.array(adverserial_image)
plt.imshow(after_im)
plt.subplot(133)
plt.title("Added noise")
noise = after_im - before_im
plt.imshow(noise)
plt.show()
