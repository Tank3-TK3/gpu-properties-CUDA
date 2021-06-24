#include <string>
#include <iostream>
#include <cuda_runtime.h>

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
void printDevProp( int i )
{
	cudaDeviceProp devProp;
	cudaGetDeviceProperties( &devProp , i );
	cout << " - Device Name: " << devProp.name << "\n";
	cout << " - Numero de revision mayoritario: " << devProp.major << "\n";
	cout << " - Numero de revision minoritario: " << devProp.minor << "\n";
	cout << " - Arquitectura: " << getType(devProp).c_str() << "\n";
	cout << "Numero de procesadores: %d\n" << devProp.multiProcessorCount;
	cout << "Cores CUDA: %d\n" << getSPcores(devProp);
	cout << "Total de memoria global:           %u\n" << devProp.totalGlobalMem;
	cout << "Total de memoria compartida por bloque: %u\n" << devProp.sharedMemPerBlock;
	cout << "Total de registros por bloque:     %d\n" << devProp.regsPerBlock;
	cout << "Tamaño del warp:                     %d\n" << devProp.warpSize;
	cout << "Pitch maximo de memoria:          %u\n" << devProp.memPitch;
	cout << "Hilos maximos por bloque:     %d\n" << devProp.maxThreadsPerBlock;
	for (int i = 0; i < 3; ++i)
		cout << "Dimension maxima %d de grid:   %d\n" << i, devProp.maxGridSize[i];
	for (int i = 0; i < 3; ++i)
		cout << "Dimension maxima %d de bloque:  %d\n" << i, devProp.maxThreadsDim[i];
	cout << "Velocidad del reloj:                    %d\n" << devProp.clockRate;
	cout << "Memoria constante total:         %u\n" << devProp.totalConstMem;
	cout << "Alineamiento de textura:             %u\n" << devProp.textureAlignment;
	cout << "Copiado y ejecucion concurrente: %s\n" << (devProp.deviceOverlap ? "Si" : "No");
	cout << "Numero de multiprocesadores:     %d\n" << devProp.multiProcessorCount;
	cout << "Timeout de ejecucion del Kernel:      %s\n" << (devProp.kernelExecTimeoutEnabled ? "Si" : "No");
	return;
}

int main( int argc , char* argv[] )
{
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount( &devCount );
	cout << "##################################################\n";
	cout << "\t  > CUDA Device Specifications <\n";
	cout << "\t     (Total CUDA devices: " << devCount << ")\n";
	// Iterate through devices
	for( int i = 0 ; i < devCount ; ++i )
	{
		cout << "##################################################\n";
		// Get device properties
		cout << "+ CUDA device: " << i << "\n";
		printDevProp( i );
		cout << "##################################################\n\n";
	}
	system( "pause" );
	return 0;
}
