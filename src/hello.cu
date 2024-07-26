#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <string.h>
#include <fstream>
#include <iostream>
#include "stbi_image.h"
#include "stbi_image_write.h"
#include <cuda_runtime.h>
#include <npp.h>

// #include <helper_cuda.h>
// #include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  // bool bVal = checkCudaCapabilities(1, 0);
  // return bVal;
  return true;
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  // try
  {
    std::string sFilename;
    char *filePath;

    // findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    // We need to load the image from disk
    if (argc == 1)
    {
      // No arguments were passed
      printf("No image data passed\n");
      exit(EXIT_FAILURE);
    }
    else
    {
      // Load the image from disk
      sFilename = std::string(argv[1]);
      filePath = argv[1];
    }

    int width, height, channels;
    unsigned char *img = stbi_load(filePath, &width, &height, &channels, 0);

    // Check if the image was loaded
    if (img == NULL)
    {
      printf("Error loading image\n");
      exit(EXIT_FAILURE);
    }

    // Convert the image to NPP format
    Npp8u *d_src = NULL;
    NppiSize oSizeROI;
    oSizeROI.width = width;
    oSizeROI.height = height;

    cudaMalloc((void **)&d_src, width * height * channels * sizeof(Npp8u));
    cudaMemcpy(d_src, img, width * height * channels * sizeof(Npp8u),
               cudaMemcpyHostToDevice);

    // Convert the image to grayscale
    Npp8u *d_dst = NULL;
    cudaMalloc((void **)&d_dst, width * height * sizeof(Npp8u));

    NppStatus npp_status;
    npp_status = nppiRGBToGray_8u_C3C1R(d_src, width * channels, d_dst, width,
                                         oSizeROI);

    if (npp_status != NPP_SUCCESS)
    {
      printf("Error converting image to grayscale\n");
      exit(EXIT_FAILURE);
    }

    // Do sobel edge detection
    /*NppStatus nppiFilterSobelHoriz_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI)*/
    Npp8u *d_sobel = NULL;
    cudaMalloc((void **)&d_sobel, width * height * sizeof(Npp8u));

    npp_status = nppiFilterSobelHoriz_8u_C1R(d_dst, width, d_sobel, width,
                                              oSizeROI);
    
    if (npp_status != NPP_SUCCESS)
    {
      printf("Error doing sobel edge detection\n");
      exit(EXIT_FAILURE);
    }


    // Copy the image back to the host
    unsigned char *sobel = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    cudaMemcpy(sobel, d_sobel, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the image to disk
    stbi_write_png("sobel.png", width, height, 1, sobel, width);

    // Free the memory
    cudaFree(d_src);
    cudaFree(d_dst);
    cudaFree(d_sobel);
    free(sobel);

    // Free the image
    stbi_image_free(img);

    return;
  }
}
