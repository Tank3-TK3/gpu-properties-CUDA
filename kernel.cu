#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>
#include <stdio.h>
#include <iostream>

using namespace std;

string getType(cudaDeviceProp devProp)
{
	string letrero = "";
	switch (devProp.major) {
	case 2: // Fermi
		letrero = "Fermi";
		break;
	case 3: // Kepler
		letrero = "Kepler";
		break;
	case 5: // Maxwell
		letrero = "Maxwell";
		break;
	case 6: // Pascal
		letrero = "Pascal";
		break;
	case 7: // Volta
		letrero = "Volta";
		break;
	default:
		letrero = "Unknown device type";
		break;
	}
	return letrero;
}

int getSPcores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if (devProp.minor == 1) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else cout << "Unknown device type\n";
		break;
	case 7: // Volta
		cores = mp * 64;
		break;
	default:
		cout << "Unknown device type\n";
		break;
	}
	return cores;
}

// Imprime las propiedades del dispositivo gráfico
void printDevProp(cudaDeviceProp devProp)
{
	cout << " - Nombre del dispositivo: " << devProp.name << "\n";
	cout << " - Numero de revision mayoritario: " << devProp.major << "\n";
	printf("Numero de revision minoritario:         %d\n", devProp.minor);
	printf("Arquitectura: %s\n", getType(devProp).c_str());
	printf("Numero de procesadores: %d\n", devProp.multiProcessorCount);
	printf("Cores CUDA: %d\n", getSPcores(devProp));
	printf("Total de memoria global:           %u\n", devProp.totalGlobalMem);
	printf("Total de memoria compartida por bloque: %u\n", devProp.sharedMemPerBlock);
	printf("Total de registros por bloque:     %d\n", devProp.regsPerBlock);
	printf("Tamaño del warp:                     %d\n", devProp.warpSize);
	printf("Pitch maximo de memoria:          %u\n", devProp.memPitch);
	printf("Hilos maximos por bloque:     %d\n", devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Dimension maxima %d de grid:   %d\n", i, devProp.maxGridSize[i]);
	for (int i = 0; i < 3; ++i)
		printf("Dimension maxima %d de bloque:  %d\n", i, devProp.maxThreadsDim[i]);
	printf("Velocidad del reloj:                    %d\n", devProp.clockRate);
	printf("Memoria constante total:         %u\n", devProp.totalConstMem);
	printf("Alineamiento de textura:             %u\n", devProp.textureAlignment);
	printf("Copiado y ejecucion concurrente: %s\n", (devProp.deviceOverlap ? "Si" : "No"));
	printf("Numero de multiprocesadores:     %d\n", devProp.multiProcessorCount);
	printf("Timeout de ejecucion del Kernel:      %s\n", (devProp.kernelExecTimeoutEnabled ? "Si" : "No"));
	return;
}

int main( int argc , char* argv[] )
{
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount( &devCount );
	cout << "\t>CUDA Device Specifications<\n";
	cout << "\t  (Total CUDA devices: " << devCount << ")\n";

	// Iterate through devices
	for( int i = 0 ; i < devCount ; ++i )
	{
		// Get device properties
		printf("\nDispositivo CUDA #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(devProp);
	}

	printf("\nPresione cualquier tecla para salir...");
	char c;
	scanf("%c", &c);

	return 0;
}
