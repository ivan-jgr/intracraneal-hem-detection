import matplotlib.pyplot as plt
from PIL import Image
from subprocess import check_output
from random import sample
from os.path import join
from matplotlib.pyplot import imsave
from torchvision import transforms

# Imagenes de prueba y transformaciones
img_path = './data/img/'
all_transforms = image = transforms.Compose([   
    transforms.Resize((224, 224)), # las imagenes originales son de tamaño 512x512
    transforms.ToTensor(), # convertir a torch.Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalización
])


def load_random_samples(n):
    """
    Arguments
    ---------
    n:  numero de ejemplos

    Returns
    -------
    imgs:   lista de torch.Tensor con las imagenes
    """

    img_names = check_output(['ls', img_path]).decode('utf8').splitlines()
    # si n > nmax, devolver n_max
    selected_images = sample(img_names, min(n, len(img_names)))
    samples_path = [join(img_path, img) for img in selected_images]
    
    imgs = []
    for sample_path in samples_path:
        x = Image.open(sample_path).convert("RGB") # leerlas con 3 canales
        x = all_transforms(x) # aplicar las transformaciones
        imgs.append(x)
    
    return imgs

def plot_images(rows, cols, images):
    """
    Arguments:
    ----------
    rows:   número de filas
    cols:   número de columnas
    images: lista de imágenes ( de tipo torch.Tensor)

    Returns:
    --------
    """

    fig, axs = plt.subplots(rows, cols, sharex='col', sharey='row', 
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    
    for i in range(rows):
        for j in range(cols):
            try:
                axs[i, j].imshow(images[i*cols + j][0, ...], cmap='gray')
            except IndexError:
                pass
    
    fig.show()