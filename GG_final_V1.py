import os
import numpy as np
import matplotlib.pyplot as plt
#import scipy.io as sio
from scipy import ndimage

def load_image(filename):
    image = plt.imread(filename)
    # input image is color png depicting grayscale, just use first plane from here on
    image = image[:, :, 1].astype(np.float64)
    # after reading and converting to float, image is between 0 and 1, we want to compute within the range 0 and 255
    image *= 255.0
    print(image.shape)
    print('image: min = ', np.min(image), ' max = ', np.max(image))
    return image

def pad_image(image):
    # compute required padding for later rotation: we assume image has same width and height
    padded_image_size = int(np.floor(image.shape[0] * np.sqrt(2)))
    # make sure padded image has odd dimensions
    if np.remainder(padded_image_size, 2) == 0:
        padded_image_size += 1

    # padding width is an integer since input size and padded size are both odd
    padding_width = int((padded_image_size - image.shape[0]) / 2)

    # place the input image into a padded version, extending with zeros
    padded_image = np.zeros((padded_image_size, padded_image_size), dtype=np.float64)
    padded_image[padding_width:image.shape[0] + padding_width,
                 padding_width:image.shape[1] + padding_width] = image

    return padded_image, padding_width

def calculate_sinogram(image, num_angles):
    image_size = image.shape[0]
    print(image_size)
    thetas = np.linspace(0., 180., num_angles, endpoint=False)
    # print(thetas.shape)
    # print(thetas)

    sinogram = np.zeros([num_angles, image_size], dtype=np.float64)

    for i, theta in enumerate(thetas):
        # print(theta)
        # rotate image by angle, counterclockwise
        imgR = ndimage.rotate(image, theta, order=1, reshape=False)

        # compute projection along columns (axis == 0 means sum is along columns!)
        projection_line = np.sum(imgR, axis=0)
        #print(projection_line.shape)

        # put projection lines into rows of sinogram (one row per discretized angle)
        sinogram[i, :] = projection_line 
        ### sinogram[nr. angles, nr. projections]

    return sinogram

def back_projection_smear(sinogram, num_angles, order=3, mode='constant'):

    reconstructed_image = np.zeros((sinogram.shape[1], sinogram.shape[1]))
    theta = np.linspace(0., 180., num_angles, endpoint=False)
    smear = np.zeros((sinogram.shape[1], sinogram.shape[1]))

    for i in range(num_angles):
            projection = sinogram[i, :]

            for x in range(sinogram.shape[1]):
                    smear[:, x] = projection[x]
        
            projection_rotated = ndimage.rotate(smear, -theta[i],order=order, mode=mode, reshape=False)
        
            reconstructed_image += projection_rotated

            # if i == 180:
            #     plt.imshow(smear, cmap='gray')
            #     plt.savefig('Smear 180')
            #     plt.show()

            #     plt.imshow(reconstructed_image, cmap='gray')
            #     plt.savefig('Summed projection 180')
            #     plt.show()

            #     plt.plot(projection)
            #     plt.savefig('Projection 180')
            #     plt.show()
    
    return reconstructed_image

def filtered_back_projection (sinogram, num_angles, f_type):

    reconstructed_image = np.zeros((sinogram.shape[1], sinogram.shape[1]))

    #plt.plot(sinogram[0])
    #plt.savefig('projection wo f')
    #plt.show()

    freqs = np.fft.fftfreq(sinogram.shape[1])
    filter = 2 * np.abs(freqs)
    if f_type == 'ramp':
        #plt.plot(filter)
        #plt.savefig('Filter')
        #plt.show()
        pass
    elif f_type == 'ram-lak':
        limit = sinogram.shape[1] // 2
        filter[limit:] = 0 
        #plt.plot(filter)
        #plt.savefig('Filter')
        #plt.show()

    fft_sino = np.fft.fft(sinogram, axis=1)
    #plt.plot(fft_sino[90])
    #plt.savefig('projection wo f')
    #plt.show()

    fft_filtered = fft_sino * filter
    #plt.plot(fft_filtered[90])
    #plt.savefig('projection w f')
    #plt.show()

    ifft_sino = np.fft.ifft(fft_filtered, axis=1)
    filtered_sinogram = np.real(ifft_sino)


    # plt.imshow(sinogram, cmap='gray')
    # plt.title('Initial sinogram')
    # plt.savefig('Init_Sino')
    # plt.show()

    # plt.imshow(filtered_sinogram, cmap='gray')
    # plt.title('Filtered sinogram')
    # plt.savefig('Filt_Sino')
    # plt.show()

    modes ={'constant','reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}

    reconstructed_image = back_projection_smear(filtered_sinogram, num_angles, order=5, mode='reflect')
     
    #plt.imshow(reconstructed_image, cmap='gray')
    #plt.savefig('Reconstructed Image')
    #plt.show()

    return reconstructed_image

def normalization(image):

  min_val = np.min(image)
  max_val = np.max(image)

  # Normalize the image to the range [0, 255]
  normalized_image = 255 * (image - min_val) / (max_val - min_val)

  # Convert the normalized image to uint8 data type
  #normalized_image = normalized_image.astype(np.uint8)

  return normalized_image

def main():

    filename = 'CTThoraxSlice257.png'

    # note: angles are always between 0 and 180 degree, more angles are not necessary since projections would be redundant
    # if num_of_angle_steps is 360, we will therefore have a discretization into 360 angle steps between 0 and 180 degree (every half angle is used)
    # you are supposed to experiment with this number to simulate lower numbers of projection lines!
    num_of_angle_steps = 360

    # load and pad image
    image = load_image(filename)
    padded_image, padding_width = pad_image(image)
    #print('padding width: ', padding_width)

    # calculate and save sinogram
    sinogram = calculate_sinogram(padded_image, num_of_angle_steps) ### sinogram[nr. angles, nr. projections]
    #sio.savemat('sinogram.mat', { 'sinogram': sinogram })

    #plt.imshow(sinogram, cmap='gray')
    #plt.title('Sinogram base')
    #plt.show()

    BP_smear = back_projection_smear(sinogram,num_of_angle_steps)

    #plt.imshow(BP_smear, cmap='gray')
    #plt.savefig('BP_final')
    #plt.show()

    GG_sino = filtered_back_projection(sinogram, num_of_angle_steps,'ramp')
    
    # plt.imshow(BP_smear, cmap='gray')
    # plt.title('Backprojection summation image')
    # plt.savefig('BP_final')
    # plt.show()

    # plt.imshow(GG_sino, cmap='gray')
    # plt.title('Filtered backprojection image')
    # plt.savefig('FBP_final')
    # plt.show()

    
    fig, axarr = plt.subplots(2, 3,figsize=(12, 8))

    axarr[0, 0].imshow(image, cmap=plt.cm.gray)
    axarr[0, 1].imshow(padded_image, cmap=plt.cm.gray)
    axarr[0, 2].imshow(sinogram, cmap=plt.cm.gray)
    axarr[1, 0].plot(sinogram[0,:])
    axarr[1, 1].imshow(BP_smear, cmap=plt.cm.gray)
    axarr[1, 2].imshow(GG_sino, cmap=plt.cm.gray)

    plt.show()


    crop_img = GG_sino[padding_width:GG_sino.shape[0]-padding_width,padding_width:GG_sino.shape[1]-padding_width]

    normalize_img = normalization(crop_img)
    #normalize_img = normalization(GG_sino)
    difference = np.abs(normalize_img - image)

    plt.imshow(difference,cmap='gray')
    plt.colorbar()
    plt.savefig('Diff')
    plt.show()


    mse = np.mean((difference)**2)
    #mse = np.mean((normalize_img-image)**2)
    rmse = np.sqrt(mse)

    print('RMSE:', rmse)

if __name__ == "__main__":
    main()
