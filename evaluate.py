from models import *

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            #t.mul_(s).add_(m)
            t = torch.add(torch.mul(t,s),m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
def single_image_inference(generator, image, noise_var = 0.1):
    #unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    unorm = invTrans
    trans1 = transforms.ToPILImage()
    transform_rand = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Lambda(lambda x: x + noise_var*(torch.rand(x.shape)-0.5)),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])
    im = image.convert('RGB').resize((256,256))
    generator = generator.cuda()
    fake = generator(transform_rand(im).view(1,3,256,256).cuda())
    fake = trans1(unorm(fake.view(3,256,256).cpu()))
    return(fake)

def crop(im, height, width):
    L = []
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            L.append(a)
    return(L)

def crop(im, height, width):
    L = []
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            L.append(a)
    return(L)

def generate_and_show(image, gen = 1):
    generator = GeneratorUNet()
    if gen:
        generator.load_state_dict(torch.load('generator_edges_shoes_L1.pth'))
    else:
        generator.load_state_dict(torch.load('generator_edges_shoes.pth'))
    image = Image.fromarray(image)
    single_image_inference(generator, image).resize((512,512)).show()
    