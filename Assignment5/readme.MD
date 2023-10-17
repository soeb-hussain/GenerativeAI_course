# 1. Photo-to-Monet Image Conversion with CycleGAN

Developed a custom residual block for feature enhancement.
Designed an encoder-decoder generator to transform images from 64x256x256 to 256x64x64.
Achieved impressive results with high-quality Monet-style image conversion and robust discriminator performance.
Attained remarkable GAN performance with G_loss of 2.57756 and Discriminator loss for photos at 0.18 and for Monet images at 0.16 after 70 epochs of training.
2. Network Architecture and Training

Created a generator with reflection padding, Conv2D, instance normalization, and ReLU activations.
Employed a straightforward four-layer discriminator with Conv2D and leakyReLU.
Trained the CycleGAN model using adversarial loss, cycle consistency loss, and identity loss with the Adam optimizer.
Successfully fine-tuned the model, resulting in the reported loss values after 70 epochs.
3. Quantifiable Achievements

Demonstrated success in photo-to-Monet image conversion with quantitative metrics, achieving high-quality results.
Leveraged cycle consistency and adversarial loss to ensure image fidelity.
Achieved notable improvements in the aesthetic and artistic quality of the generated Monet-style paintings, validated by low GAN losses post 70 epochs.
