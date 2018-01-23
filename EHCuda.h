#ifndef EH_CUDA_H
#define EH_CUDA_H

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <x86intrin.h>

#ifdef FFT
#include <cufft.h>
#endif

#ifdef DEBUG
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define STB_SPRINTF_IMPLEMENTATION
#include "stb_sprintf.h"
#endif

typedef int8_t s08;
typedef int16_t s16;
typedef int32_t s32;
typedef int64_t s64;

typedef uint8_t u08;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef bool b32;

typedef float f32;
typedef double f64;

typedef size_t memptr;

typedef float2 fComplex;

#define s08_max INT8_MAX
#define s16_max INT16_MAX
#define s32_max INT32_MAX
#define s64_max INT64_MAX

#define s08_min INT8_MIN
#define s16_min INT16_MIN
#define s32_min INT32_MIN
#define s64_min INT64_MIN

#define u08_max UINT8_MAX
#define u16_max UINT16_MAX
#define u32_max UINT32_MAX
#define u64_max UINT64_MAX

#define f32_max MAXFLOAT

#define Min(x, y) (x < y ? x : y)
#define Max(x, y) (x > y ? x : y)

#define u08_n (u08_max + 1)

#define Square(x) (x * x)

#define Pow10(x) (IntPow(10, x))
#define Pow2(N) (1 << N)

#define ArrayCount(array) (sizeof(array) / sizeof(array[0]))

#define ArgCount argc
#define ArgBuffer argv
#define Main s32 main()
#define MainArgs s32 main(s32 ArgCount, const char *ArgBuffer[])
#define OldMain s32 oldmain(s32 ArgCount, const char *ArgBuffer[])
#define EndMain return(0)

#ifdef DEBUG
#define CUDA_ERROR_CHECK
#endif

#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    cudaError err = cudaGetLastError();
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = cudaDeviceSynchronize();
    if( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString( err ) );
        exit( -1 );
    }
#endif

    return;
}

#define CudaKernel __global__ void
#define CudaFunction __device__
#define CudaMalloc(ptr, size) CudaSafeCall(cudaMalloc((void **)&ptr, size))
#define CudaMallocManaged(ptr, size) CudaSafeCall(cudaMallocManaged((void **)&ptr, size))
#define MyIndex ((blockDim.x * blockIdx.x) + threadIdx.x) 
#define GridSize (blockDim.x * gridDim.x)

#define LaunchCudaKernel_Simple(Kernel, ...) Kernel<<<64, 64>>>(__VA_ARGS__); \
	CudaCheckError()
#define LaunchCudaKernel(Kernel, ThreadBlockStruct, ...) Kernel<<<ThreadBlockStruct.nBlocks, ThreadBlockStruct.threadsPerBlock>>>(__VA_ARGS__); \
	CudaCheckError()
#define LaunchCudaKernel_Simple_Stream(Kernel, Stream, ...) Kernel<<<64, 64, 0, Stream>>>(__VA_ARGS__); \
	CudaCheckError()
#define LaunchCudaKernel_Stream(Kernel, Grid, Threads, Stream, ...) Kernel<<<Grid, Threads, 0, Stream>>>(__VA_ARGS__); \
	CudaCheckError()
#define CudaCopyFromDevice(HostData, DeviceData, threadBlockStruct) CudaSafeCall(cudaMemcpy(HostData, DeviceData, threadBlockStruct.nBytes, cudaMemcpyDeviceToHost))
#define CudaCopyToDevice(HostData, DeviceData, threadBlockStruct) CudaSafeCall(cudaMemcpy(DeviceData, HostData, threadBlockStruct.nBytes, cudaMemcpyHostToDevice))

#define OneDCudaLoop(index, endCount) for ( u32 index = MyIndex; \
		index < (endCount); \
		index += GridSize )
	
#include <pthread.h>

#define ThreadFence __asm__ volatile("" ::: "memory")
#define FenceIn(x) ThreadFence; \
	x; \
	ThreadFence

typedef pthread_t thread;
typedef pthread_mutex_t mutex;
typedef pthread_cond_t cond;

typedef volatile u32 threadSig;

#define CreateThread(x) thread *x
#define CreateMutex(x) mutex *x
#define CreateCond(x) cond *x

#ifdef DEBUG
#define InitialiseMutex(x) *x = (mutex)PTHREAD_MUTEX_INITIALIZER
#define InitialiseCond(x) *x = (cond)PTHREAD_COND_INITIALIZER
#else
#define InitialiseMutex(x) *x = PTHREAD_MUTEX_INITIALIZER
#define InitialiseCond(x) *x = PTHREAD_COND_INITIALIZER
#endif

#define LaunchThread(thread, func, dataIn) pthread_create(thread, NULL, func, dataIn);
#define WaitForThread(x) pthread_join(*x, NULL)

#define LockMutex(x) pthread_mutex_lock(x)
#define UnlockMutex(x) pthread_mutex_unlock(x)
#define WaitOnCond(cond, mutex) pthread_cond_wait(cond, mutex) 
#define SignalCondition(x) pthread_cond_broadcast(x)

typedef cudaStream_t stream;

#define CreateStream(x) stream *x

#ifdef DEBUG
#include <assert.h>
#define Assert(x) assert(x)
#else
#define Assert(x)
#endif

#define KiloByte(x) 1024*x
#define MegaByte(x) 1024*KiloByte(x)
#define GigaByte(x) 1024*MegaByte(x)

#include "zlib.h"
#define CHUNK KiloByte(256)

#define Default_Memory_Alignment_Pow2 4

struct
memory_arena
{
	u08 *base;
	memptr currentSize;
	memptr maxSize;
};

void
CreateMemoryArena_(memory_arena *arena, memptr size, memptr alignment_pow2 = Default_Memory_Alignment_Pow2)
{
	posix_memalign((void **)&arena->base, Pow2(alignment_pow2), size);
	arena->currentSize = 0;
	arena->maxSize = size;
}

#define CreateMemoryArena(arena, size, ...) CreateMemoryArena_(&arena, size, ##__VA_ARGS__)
#define CreateMemoryArenaP(arena, size, ...) CreateMemoryArena_(arena, size, ##__VA_ARGS__)

void
ResetMemoryArena_(memory_arena *arena)
{
	arena->currentSize = 0;
}

#define ResetMemoryArena(arena) ResetMemoryArena_(&arena)
#define ResetMemoryArenaP(arena) ResetMemoryArena_(arena)

void
FreeMemoryArena_(memory_arena *arena)
{
	free(arena->base);
}

#define FreeMemoryArena(arena) FreeMemoryArena_(&arena)
#define FreeMemoryArenaP(arena) FreeMemoryArena_(arena)

memptr
GetAlignmentPadding(memptr base, u32 alignment_pow2)
{
	u64 alignment = (u64)Pow2(alignment_pow2);
	memptr result = ((base + alignment - 1) & ~(alignment - 1)) - base;

	return(result);
}

u32
AlignUp(u32 x, u32 alignment_pow2)
{
	u32 alignment_m1 = Pow2(alignment_pow2) - 1;
	u32 result = (x + alignment_m1) & ~alignment_m1;

	return(result);
}

void *
PushSize_(memory_arena *arena, memptr size, u32 alignment_pow2 = Default_Memory_Alignment_Pow2)
{
	memptr padding = GetAlignmentPadding((memptr)(arena->base + arena->currentSize), alignment_pow2);
	
	void *result;
	if((size + arena->currentSize + padding + sizeof(memptr)) > arena->maxSize)
	{
		result = 0;
	}
	else
	{
		result = arena->base + arena->currentSize + padding;
		arena->currentSize += (size + padding + sizeof(memptr));
		
		*((memptr *)(arena->base + arena->currentSize - sizeof(memptr))) = (size + padding);
	}
	
	return(result);
}

void
FreeLastPush_(memory_arena *arena)
{
	if (arena->currentSize)
	{
		memptr sizeToRemove = *((memptr *)(arena->base + arena->currentSize - sizeof(memptr)));
		arena->currentSize -= (sizeToRemove + sizeof(memptr));
	}
}

#define PushStruct(arena, type, ...) (type *)PushSize_(&arena, sizeof(type), ##__VA_ARGS__)
#define PushArray(arena, type, n, ...) (type *)PushSize_(&arena, sizeof(type) * n, ##__VA_ARGS__)
#define PushStructP(arena, type, ...) (type *)PushSize_(arena, sizeof(type), ##__VA_ARGS__)
#define PushArrayP(arena, type, n, ...) (type *)PushSize_(arena, sizeof(type) * n, ##__VA_ARGS__)

#define FreeLastPush(arena) FreeLastPush_(&arena)
#define FreeLastPushP(arena) FreeLastPush_(arena)

memory_arena *
PushSubArena_(memory_arena *mainArena, memptr size, memptr alignment_pow2 = Default_Memory_Alignment_Pow2)
{
	memory_arena *subArena = PushStructP(mainArena, memory_arena, alignment_pow2);
	subArena->base = PushArrayP(mainArena, u08, size, alignment_pow2);
	subArena->currentSize = 0;
	subArena->maxSize = size;

	return(subArena);
}

#define PushSubArena(arena, size, ...) PushSubArena_(&arena, size, ##__VA_ARGS__)
#define PushSubArenaP(arena, size, ...) PushSubArena_(arena, size, ##__VA_ARGS__)

void *
AllocCallBack(void *arena, u32 items, u32 size)
{
	return (void *)(PushArrayP((memory_arena *)arena, u08, items * size));	
}

void
FreeCallBack(void *arena, void *ptr)
{
	
}

s32
Compress(memory_arena *arena, u08 *source, u08 *dest, u32 nBytesIn, u32 *nBytesOut, u32 level)
{
    ResetMemoryArenaP(arena);

    s32 ret, flush;
    u32 have;
    z_stream strm;
    u08 *in;
    
    strm.zalloc = AllocCallBack;
    strm.zfree = FreeCallBack;
    strm.opaque = (void *)arena;
    ret = deflateInit(&strm, level);
    if (ret != Z_OK)
        return ret;

    u32 ChunkCounter = 0;
    s32 nBytesLeft = (s32)nBytesIn;
    u32 nBytesOut_local = 0;
    do {
	in = source + (ChunkCounter++ * CHUNK);
        nBytesLeft -= CHUNK;
	strm.avail_in = nBytesLeft >= 0 ? CHUNK : (nBytesLeft + CHUNK); 

        flush = nBytesLeft < 0 ? Z_FINISH : Z_NO_FLUSH;
        strm.next_in = in;

        do {
            strm.avail_out = CHUNK;
            strm.next_out = dest;
	    ret = deflate(&strm, flush);    
            Assert(ret != Z_STREAM_ERROR);  
            have = CHUNK - strm.avail_out;
	    dest += have;
	    nBytesOut_local += have;

        } while (strm.avail_out == 0);

    } while (flush != Z_FINISH);

    *nBytesOut = nBytesOut_local;
    (void)deflateEnd(&strm);
    
    return Z_OK;
}

s32
Decompress(memory_arena *arena, u08 *source, u08 *dest, u32 nBytesIn, u32 *nBytesOut)
{
    ResetMemoryArenaP(arena);

    s32 ret;
    u32 have;
    z_stream strm;
    
    u08 *in;
    
    strm.zalloc = AllocCallBack;
    strm.zfree = FreeCallBack;
    strm.opaque = (void *)arena;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;
    ret = inflateInit(&strm);
    if (ret != Z_OK)
        return ret;

    u32 ChunkCounter = 0;
    s32 nBytesLeft = (s32)nBytesIn;
    u32 nBytesOut_local = 0;
    do {
	if (nBytesLeft < 0)
		break;
	    
	in = source + (ChunkCounter++ * CHUNK);
        nBytesLeft -= CHUNK;
	strm.avail_in = nBytesLeft >= 0 ? CHUNK : (nBytesLeft + CHUNK); 

        strm.next_in = in;

        do {
            strm.avail_out = CHUNK;
            strm.next_out = dest;
            ret = inflate(&strm, Z_NO_FLUSH);
            switch (ret) {
            case Z_NEED_DICT:
                ret = Z_DATA_ERROR;     
            case Z_DATA_ERROR:
            case Z_MEM_ERROR:
                (void)inflateEnd(&strm);
                return ret;
            }
            have = CHUNK - strm.avail_out;
            dest += have;
	    nBytesOut_local += have;
        } while (strm.avail_out == 0);

    } while (ret != Z_STREAM_END);

    *nBytesOut = nBytesOut_local;
    (void)inflateEnd(&strm);
    
    return ret == Z_STREAM_END ? Z_OK : Z_DATA_ERROR;
}

/* report a zlib or i/o error */
#if 0
void zerr(int ret)
{
    fputs("zpipe: ", stderr);
    switch (ret) {
    case Z_ERRNO:
        if (ferror(stdin))
            fputs("error reading stdin\n", stderr);
        if (ferror(stdout))
            fputs("error writing stdout\n", stderr);
        break;
    case Z_STREAM_ERROR:
        fputs("invalid compression level\n", stderr);
        break;
    case Z_DATA_ERROR:
        fputs("invalid or incomplete deflate data\n", stderr);
        break;
    case Z_MEM_ERROR:
        fputs("out of memory\n", stderr);
        break;
    case Z_VERSION_ERROR:
        fputs("zlib version mismatch!\n", stderr);
    }
}
#endif

enum
file_packing_order
{
	ZTC,
	ZCT,
	TZC,
	TCZ,
	CZT,
	CTZ,
};

struct
SSIF_channel_name
{
	char name[32];
};

struct
SSIF_file_header_packed
{
	char name[64];
	u32 width;
	u32 height;
	u32 depth;
	u32 timepoints;
	u32 channels;
	u32 bytesPerPixel;
	u32 packingOrder;
		
}__attribute__((packed));

struct
SSIF_file_header
{
	char name[64];
	u32 width;
	u32 height;
	u32 depth;
	u32 timepoints;
	u32 channels;
	u32 bytesPerPixel;
	file_packing_order packingOrder;

	SSIF_channel_name *channelNames;
};

struct
SSIF_image_file_location
{
	u64 base;
	u32 nBytes;
};

struct
SSIF_image_file_directory
{
	SSIF_image_file_location *locations;
};

union
image_file_coords
{
	struct
	{
		u32 z;
		u32 t;
		u32 c;
	};
	u32 E[3];
};

struct
volatile_image_file_coords
{
	volatile u64 c_t_z;
};

struct
SSIF_file
{
	FILE *file;
	SSIF_file_header *header;
};

struct
SSIF_file_for_reading
{
	SSIF_file *SSIFfile;
	SSIF_image_file_directory *directory;
};

struct
SSIF_file_for_writing
{
	SSIF_file *SSIFfile;
	volatile_image_file_coords currentCoords;
};

u32
AreNullTerminatedStringsEqual(char *string1, char *string2)
{
	u32 result;
	do
	{
		result = (*string1 == *string2);
		++string2;
	} while(result && (*(string1++) != '\0'));

	return(result);
}

u32
CopyNullTerminatedString(char *source, char *dest)
{
	u32 stringLength = 0;

	while(*source != '\0')
	{
		*(dest++) = *(source++);
		++stringLength;
	}
	*dest = '\0';

	return(stringLength);
}

u32
CopySSIFHeaderName(char *source, char *dest, u32 currentLength = 0)
{
	u32 index = currentLength;
	for (	;
		index < 64 && *source != '\0';
		++index )
	{
		*(dest++) = *(source++);
	}

	return(index);
}

u32
CopySSIFHeader(SSIF_file_header *source, SSIF_file_header *dest, char *destName)
{
	u32 nameLength = CopySSIFHeaderName(destName, dest->name);

	dest->width = source->width;
	dest->height = source->height;
	dest->depth = source->depth;
	dest->timepoints = source->timepoints;
	dest->channels = source->channels;
	dest->bytesPerPixel = source->bytesPerPixel;
	dest->packingOrder = source->packingOrder;

	dest->channelNames = source->channelNames;

	return(nameLength);
}

void
PackSSIFHeader(SSIF_file_header *header, SSIF_file_header_packed *packed)
{
	CopySSIFHeaderName(header->name, packed->name);
	
	packed->width = header->width;
	packed->height = header->height;
	packed->depth = header->depth;
	packed->timepoints = header->timepoints;
	packed->channels = header->channels;
	packed->bytesPerPixel = header->bytesPerPixel;
	packed->packingOrder = (u32)(s32)header->packingOrder;
}

void
UnpackSSIFHeader(SSIF_file_header *header, SSIF_file_header_packed *packed)
{
	CopySSIFHeaderName(packed->name, header->name);
	
	header->width = packed->width;
	header->height = packed->height;
	header->depth = packed->depth;
	header->timepoints = packed->timepoints;
	header->channels = packed->channels;
	header->bytesPerPixel = packed->bytesPerPixel;
	header->packingOrder = (file_packing_order)(s32)packed->packingOrder;
}

void
AllocateSSIFImageFileLocations(memory_arena *arena, SSIF_file_for_reading *SSIFfile_reading)
{
	u32 nImages = SSIFfile_reading->SSIFfile->header->depth * SSIFfile_reading->SSIFfile->header->timepoints * SSIFfile_reading->SSIFfile->header->channels;
	SSIFfile_reading->directory->locations = PushArrayP(arena, SSIF_image_file_location, nImages);
}

SSIF_image_file_location *
LookupSSIFImageLocation(SSIF_file_for_reading *SSIFfile_reading, u32 z, u32 t, u32 c)
{
	u32 index;
	
	switch (SSIFfile_reading->SSIFfile->header->packingOrder)
	{
		case ZTC:
			{
				index = (SSIFfile_reading->SSIFfile->header->depth * SSIFfile_reading->SSIFfile->header->timepoints * c) + (SSIFfile_reading->SSIFfile->header->depth * t) + z;
			} break;
		
		case ZCT:
			{
				index = (SSIFfile_reading->SSIFfile->header->depth * SSIFfile_reading->SSIFfile->header->channels * t) + (SSIFfile_reading->SSIFfile->header->depth * c) + z;
			} break;
		
		case TZC:
			{
				index = (SSIFfile_reading->SSIFfile->header->depth * SSIFfile_reading->SSIFfile->header->timepoints * c) + (SSIFfile_reading->SSIFfile->header->timepoints * z) + t;
			} break;
		
		case TCZ:
			{
				index = (SSIFfile_reading->SSIFfile->header->timepoints * SSIFfile_reading->SSIFfile->header->channels * z) + (SSIFfile_reading->SSIFfile->header->timepoints * c) + t;
			} break;

		case CZT:
			{
				index = (SSIFfile_reading->SSIFfile->header->depth * SSIFfile_reading->SSIFfile->header->channels * t) + (SSIFfile_reading->SSIFfile->header->channels * z) + c;
			} break;

		case CTZ:
			{
				index = (SSIFfile_reading->SSIFfile->header->timepoints * SSIFfile_reading->SSIFfile->header->channels * z) + (SSIFfile_reading->SSIFfile->header->channels * t) + c;
			} break;
	}

	return SSIFfile_reading->directory->locations + index;
}

void
GetSSIFImageAtLocation(SSIF_file *SSIFfile, SSIF_image_file_location *loc, u08 *buff)
{
	fseek(SSIFfile->file, loc->base, SEEK_SET);
	fread(buff, loc->nBytes, 1, SSIFfile->file);
}

SSIF_file_for_reading *
OpenSSIFFileForReading(memory_arena *arena, char *fileName)
{
	SSIF_file_for_reading *SSIFfile_reading = PushStructP(arena, SSIF_file_for_reading);
	SSIF_file *SSIFfile = PushStructP(arena, SSIF_file);
	SSIFfile_reading->SSIFfile = SSIFfile;
	SSIFfile->file = fopen(fileName, "rb");
	
	SSIFfile->header = PushStructP(arena, SSIF_file_header);
	SSIF_file_header_packed *headerPacked = PushStructP(arena, SSIF_file_header_packed);
	fread(headerPacked, sizeof(SSIF_file_header_packed), 1, SSIFfile->file);
	UnpackSSIFHeader(SSIFfile->header, headerPacked);
	FreeLastPushP(arena); // headerPacked

	SSIFfile->header->channelNames = PushArrayP(arena, SSIF_channel_name, SSIFfile->header->channels);
	for (	u32 iChannel = 0;
		iChannel < SSIFfile->header->channels;
		++iChannel)
	{
		fread((SSIFfile->header->channelNames + iChannel)->name, sizeof(SSIF_channel_name), 1, SSIFfile->file);
	}

	SSIFfile_reading->directory = PushStructP(arena, SSIF_image_file_directory);
	AllocateSSIFImageFileLocations(arena, SSIFfile_reading);

	u64 currentLocation = (u64)(sizeof(SSIF_file_header_packed) + SSIFfile->header->channels * sizeof(SSIF_channel_name));
	for (	u32 iChannel = 0, index = 0;
		iChannel < SSIFfile->header->channels;
		++iChannel )
	{
		for (	u32 iTime = 0;
			iTime < SSIFfile->header->timepoints;
			++iTime )
		{
			for (	u32 iZ = 0;
				iZ < SSIFfile->header->depth;
				++iZ, ++index )
			{
				u32 nBytes;
				fread(&nBytes, sizeof(u32), 1, SSIFfile->file);
				fseek(SSIFfile->file, nBytes, SEEK_CUR);
				currentLocation += sizeof(u32);

				SSIF_image_file_location *loc = SSIFfile_reading->directory->locations + index;
				loc->base = currentLocation;
				loc->nBytes = nBytes;

				currentLocation += nBytes;
			}
		}
	}

	return(SSIFfile_reading);
}

SSIF_file_for_writing *
OpenSSIFFileForWriting(memory_arena *arena, SSIF_file_header *header, char *fileName)
{
	SSIF_file_for_writing *SSIFfile_writing = PushStructP(arena, SSIF_file_for_writing);
	SSIF_file *SSIFfile = PushStructP(arena, SSIF_file);
	SSIFfile->header = header;
	SSIFfile_writing->SSIFfile = SSIFfile;
	SSIFfile->file = fopen(fileName, "wb");

	SSIF_file_header_packed *packedHeader = PushStructP(arena, SSIF_file_header_packed);
	PackSSIFHeader(header, packedHeader);

	fwrite(packedHeader, sizeof(SSIF_file_header_packed), 1, SSIFfile->file);

	FreeLastPushP(arena); // packedHeader

	for (	u32 iChannel = 0;
		iChannel < header->channels;
		++iChannel )
	{
		fwrite((header->channelNames + iChannel)->name, sizeof(SSIF_channel_name), 1, SSIFfile->file);
	}

	SSIFfile_writing->currentCoords.c_t_z = 0;

	return(SSIFfile_writing);
}

void
ReadImageFromSSIFFile(memory_arena *zlibArena, SSIF_file_for_reading *SSIFfile_reading, u32 z, u32 t, u32 c, u08 *imageBuffer, u08 *compressedImageBuffer)
{
	SSIF_image_file_location *loc = LookupSSIFImageLocation(SSIFfile_reading, z, t, c);
	GetSSIFImageAtLocation(SSIFfile_reading->SSIFfile, loc, compressedImageBuffer);

	u32 nBytesDecompressedImage;
	Decompress(zlibArena, compressedImageBuffer, imageBuffer, loc->nBytes, &nBytesDecompressedImage);
}

u64
CompressCoords(image_file_coords *coords)
{
	return((((u64)coords->c << 32) & 0xffff00000000) | (((u64)coords->t << 16) & 0xffff0000) | ((u64)coords->z & 0xffff));  
}

u32
ThisIsNextImageToWrite(volatile_image_file_coords currentCoords, image_file_coords *coordsToWrite)
{
	u64 expected = CompressCoords(coordsToWrite);

	u32 result = __atomic_compare_exchange(&currentCoords.c_t_z, &expected, &expected, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

	return(result);
}

image_file_coords
LinearIndexToImageCoords(u32 index, SSIF_file_header *header)
{
	image_file_coords coords;

	u32 *first, *second, *third;
	u32 firstN, secondN;

	switch (header->packingOrder)
	{
		case ZTC:
			{
				first = &coords.z;
				second = &coords.t;
				third = &coords.c;

				firstN = header->depth;
				secondN = header->timepoints;
			} break;
		
		case ZCT:
			{
				first = &coords.z;
				second = &coords.c;
				third = &coords.t;

				firstN = header->depth;
				secondN = header->channels;
			} break;	
		
		case TZC:
			{
				first = &coords.t;
				second = &coords.z;
				third = &coords.c;

				firstN = header->timepoints;
				secondN = header->depth;
			} break;	
		
		case TCZ:
			{
				first = &coords.t;
				second = &coords.c;
				third = &coords.z;

				firstN = header->timepoints;
				secondN = header->channels;
			} break;	
		
		case CZT:
			{
				first = &coords.c;
				second = &coords.z;
				third = &coords.t;

				firstN = header->channels;
				secondN = header->depth;
			} break;	
		
		case CTZ:
			{
				first = &coords.c;
				second = &coords.t;
				third = &coords.z;

				firstN = header->channels;
				secondN = header->timepoints;
			} break;
	}

	*third = index / (firstN * secondN);
	index -= (*third * firstN * secondN);

	*second = index / firstN;
	index -= (*second * firstN);
	
	*first = index;

	return(coords);
}

void
IncrementCoords(image_file_coords *coords, SSIF_file_header *header)
{
	u32 *first, *second, *third;
	u32 firstN, secondN;

	switch (header->packingOrder)
	{
		case ZTC:
			{
				first = &coords->z;
				second = &coords->t;
				third = &coords->c;

				firstN = header->depth;
				secondN = header->timepoints;
			} break;
		
		case ZCT:
			{
				first = &coords->z;
				second = &coords->c;
				third = &coords->t;

				firstN = header->depth;
				secondN = header->channels;
			} break;	
		
		case TZC:
			{
				first = &coords->t;
				second = &coords->z;
				third = &coords->c;

				firstN = header->timepoints;
				secondN = header->depth;
			} break;	
		
		case TCZ:
			{
				first = &coords->t;
				second = &coords->c;
				third = &coords->z;

				firstN = header->timepoints;
				secondN = header->channels;
			} break;	
		
		case CZT:
			{
				first = &coords->c;
				second = &coords->z;
				third = &coords->t;

				firstN = header->channels;
				secondN = header->depth;
			} break;	
		
		case CTZ:
			{
				first = &coords->c;
				second = &coords->t;
				third = &coords->z;

				firstN = header->channels;
				secondN = header->timepoints;
			} break;
	}

	++(*first);
	if (*first == firstN)
	{
		*first = 0;
		++(*second);
		if (*second == secondN)
		{
			*second = 0;
			++(*third);
		}
	}
}

u32
WriteImageToSSIFFile(memory_arena *zlibArena, SSIF_file_for_writing *SSIFfile_writing, u08 *imageBuffer, u32 nBytesImage, u08 *compBuffer, image_file_coords *coordsToWrite, u32 zlibCompressionLevel = Z_BEST_COMPRESSION)
{
	u32 result = 0;

	if (ThisIsNextImageToWrite(SSIFfile_writing->currentCoords, coordsToWrite))
	{
		result = 1;
		u32 nBytesCompImage;
		Compress(zlibArena, imageBuffer, compBuffer, nBytesImage, &nBytesCompImage, zlibCompressionLevel);
		fwrite(&nBytesCompImage, sizeof(u32), 1, SSIFfile_writing->SSIFfile->file);
		fwrite(compBuffer, nBytesCompImage, 1, SSIFfile_writing->SSIFfile->file);

		image_file_coords tmp = *coordsToWrite;
		IncrementCoords(&tmp, SSIFfile_writing->SSIFfile->header);
		u64 compressedCoords = CompressCoords(&tmp);
		u64 currentcompressedCoords = SSIFfile_writing->currentCoords.c_t_z;

		__atomic_compare_exchange(&SSIFfile_writing->currentCoords.c_t_z, &currentcompressedCoords, &compressedCoords, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
	}

	return(result);
}

u32
GetSSIFPixelsPerImage(SSIF_file *SSIFfile)
{
	return SSIFfile->header->width * SSIFfile->header->height;
}

u32
GetSSIFBytesPerPixel(SSIF_file *SSIFfile)
{
	return SSIFfile->header->bytesPerPixel;
}

u32
GetSSIFBytesPerImage(SSIF_file *SSIFfile)
{
	return GetSSIFPixelsPerImage(SSIFfile) * SSIFfile->header->bytesPerPixel;
}

void
CloseSSIFFile(SSIF_file *SSIFfile)
{
	fclose(SSIFfile->file);
}

inline
u32
IntPow(u32 base, u32 pow)
{
    	u32 result = 1;
    
    	for(u32 index = 0;
         	index < pow;
         	++index)
    	{
        	result *= base;
    	}
    
    	return(result);
}

inline
u32
StringLength(char *string)
{
    	u32 length = 0;
    
    	while(*string++ != '\0') ++length;
    	
    	return(length);
}

struct
string_to_int_result
{
    	u32 integerValue;
    	u32 numDigits;
};

inline
string_to_int_result
StringToInt(char *string)
{
    	string_to_int_result result;
    
    	u32 strLen = 1;
    	while(*++string != '\0') ++strLen;
    
    	result.integerValue = 0;
    	result.numDigits = strLen;
    	u32 pow = 1;
    
    	while(--strLen > 0)
    	{
        	result.integerValue += (*--string - '0') * pow;
		pow *= 10;
	}
	result.integerValue += (*--string - '0') * pow;
    
	return(result);
}

u32
SIMDTestEquali(__m128i a, __m128i b)
{
	__m128i neq = _mm_xor_si128(a, b);
	return(_mm_test_all_zeros(neq, neq));
}

__m128i
CountSetBitsPerByte(__m128i x)
{
	__m128i countMask = _mm_set1_epi8(0x0F);
	__m128i countTable = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
	__m128i count0 = _mm_shuffle_epi8(countTable, _mm_and_si128(x, countMask));
	__m128i count1 = _mm_shuffle_epi8(countTable, _mm_and_si128(_mm_srli_epi16(x, 4), countMask));

	return(_mm_add_epi8(count0, count1));
}

u32
IntDivideCeil(u32 x, u32 y)
{
	u32 result = (x + y - 1) / y;
	return(result);
}

CudaFunction
u32
IntDivideCeil(u32 x, u32 y)
{
	u32 result = (x + y - 1) / y;
	return(result);
}

CudaFunction
int3
OneDToThreeD(u32 oneD, dim3 dims)
{
	int3 result;
	u32 planeSize = dims.x * dims.y;
	result.z = oneD / planeSize;
	oneD -= (result.z * planeSize);
	result.y = oneD / dims.y;
	oneD -= (result.y * dims.y);
	result.x = oneD;

	return(result);
}

CudaFunction
u32
ThreeDToOneD(dim3 position, dim3 dims)
{
	u32 result = (position.z * dims.x * dims.y) + (position.y * dims.x) + position.x;

	return(result);
}

CudaFunction
dim3
Dim3Add(dim3 u, dim3 v)
{
	dim3 result;
	result.x = u.x + v.x;
	result.y = u.y + v.y;
	result.z = u.z + v.z;

	return(result);
}

CudaFunction
int3
Dim3Subtract(dim3 u, dim3 v)
{
	int3 result;
	result.x = u.x - v.x;
	result.y = u.y - v.y;
	result.z = u.z - v.z;

	return(result);
}

CudaFunction
dim3
Dim3Hadamard(dim3 u, dim3 v)
{
	dim3 result;
	result.x = u.x * v.x;
	result.y = u.y * v.y;
	result.z = u.z * v.z;

	return(result);
}

CudaFunction
dim3
Dim3Divide(dim3 u, dim3 v)
{
	dim3 result;
	result.x = u.x / v.x;
	result.y = u.y / v.y;
	result.z = u.z / v.z;

	return(result);
}

CudaFunction
dim3
Dim3Divide_Ceil(dim3 u, dim3 v)
{
	dim3 result;
	
	result.x = IntDivideCeil(u.x, v.x);
	result.y = IntDivideCeil(u.y, v.y);
	result.z = IntDivideCeil(u.z, v.z);

	return(result);
}

CudaFunction
dim3
Dim3Half_Ceil(dim3 v)
{
	dim3 twos;
	twos.x = 2;
	twos.y = 2;
	twos.z = 2;
	dim3 result = Dim3Divide_Ceil(v, twos);

	return(result);
}

CudaFunction
u32
Int3VectorLengthSq(int3 v)
{
	return((u32)(Square(v.x) + Square(v.y) + Square(v.z)));
}

CudaFunction
dim3
Int3ToDim3(int3 a)
{
	dim3 result;
	result.x = (u32)a.x;
	result.y = (u32)a.y;
	result.z = (u32)a.z;

	return(result);
}

u32
Dim3N(dim3 u)
{
	u32 result = u.x * u.y * u.z;

	return(result);
}

CudaFunction
u32
Dim3N(dim3 u)
{
	u32 result = u.x * u.y * u.z;

	return(result);
}

CudaFunction
u32
FloatFlip(u32 f)
{
	u32 mask;
	asm("shr.u32 %0, %1, 31;" : "=r"(mask) : "r"(f));
	asm("sub.s32 %0, 0, %0;" : "+r"(mask));
	asm("or.b32 %0, %0, -2147483648;" : "+r"(mask));
	asm("xor.b32 %0, %0, %1;" : "+r"(mask) : "r"(f));
	return(mask);
}

CudaFunction
u32
IFloatFlip(u32 f)
{
	u32 mask;
	asm("shr.u32 %0, %1, 31;" : "=r"(mask) : "r"(f));
	asm("add.s32 %0, %0, -1;" : "+r"(mask));
	asm("or.b32 %0, %0, -2147483648;" : "+r"(mask));
	asm("xor.b32 %0, %0, %1;" : "+r"(mask) : "r"(f));
	return(mask);
}

CudaFunction
u32
OneDMapToReflectionGrid(u32 index, u32 dimSize)
{
	u32 result;

	if (dimSize > 1)
	{
		u32 interval = (2 * (dimSize - 1));
		result = index % interval;
		if (result >= dimSize)
		{
			result = interval - result;
		}
	}
	else
	{
		result = 0;
	}

	return(result);
}

CudaFunction
dim3
CoordinateMap_Reflection(int3 coord, dim3 dims)
{
	coord.x = coord.x < 0 ? -coord.x : coord.x;
	coord.y = coord.y < 0 ? -coord.y : coord.y;
	coord.z = coord.z < 0 ? -coord.z : coord.z;
	coord.x = OneDMapToReflectionGrid(coord.x, dims.x);
	coord.y = OneDMapToReflectionGrid(coord.y, dims.y);
	coord.z = OneDMapToReflectionGrid(coord.z, dims.z);

	dim3 result = Int3ToDim3(coord);

	return(result);
}

struct
three_d_volume
{
	dim3 dims;
	u32 size;
};

void
CreateThreeDVol_FromDim(three_d_volume *vol, dim3 dims)
{
	vol->dims = dims;
	vol->size = dims.x * dims.y * dims.z;
}
	
void
CreateThreeDVol_FromInt(three_d_volume *vol, u32 x, u32 y, u32 z)
{
	dim3 dims;
	dims.x = x;
	dims.y = y;
	dims.z = z;

	vol->dims = dims;
	vol->size = x * y * z;
}

void
CreateThreeDVol_FromString(three_d_volume *vol, const char **args)
{
	string_to_int_result local_x = StringToInt((char *)args[0]);
	string_to_int_result local_y = StringToInt((char *)args[1]);
	string_to_int_result local_z = StringToInt((char *)args[2]);

	CreateThreeDVol_FromInt(vol, local_x.integerValue, local_y.integerValue, local_z.integerValue);
}

struct
image_data
{
	u08 *image;
	memptr nBytes;
	three_d_volume vol;
};

void
CreateImageData_FromInt(image_data *imageData, u32 x, u32 y, u32 z, u32 bytesPerPixel = 1)
{
	CreateThreeDVol_FromInt(&imageData->vol, x, y, z);
	imageData->nBytes = imageData->vol.size * bytesPerPixel;
	CudaMallocManaged(imageData->image, imageData->nBytes);
}

void
CreateImageData_FromDim3(image_data *imageData, dim3 dims, u32 bytesPerPixel = 1)
{
	CreateImageData_FromInt(imageData, dims.x, dims.y, dims.z, bytesPerPixel);
}
	
void
CreateImageData_FromString(image_data *imageData, const char **args)
{
	CreateThreeDVol_FromString(&imageData->vol, args);
	imageData->nBytes = imageData->vol.size * sizeof(u08);
	CudaMallocManaged(imageData->image, imageData->nBytes);
}

void
FreeImageData(image_data *imageData)
{
	cudaFree(imageData->image);
}

enum
image_track
{
	track_depth,
	track_timepoint,
	track_channel,
};

struct image_loader
{
	SSIF_file_for_reading *SSIFfile;
	u32 channel;
	u32 timepoint;
	u32 depth;
	image_track track;
	u08 *compressedImageBuffer;
	memory_arena *zlibArena;
};

union
dim_range
{
	struct
	{
		s32 start;
		s32 end;
	};
	int2 E;
};

struct
program_arguments
{
	char *inputFile;
	
	union
	{
		dim3 histDims;
		dim3 ballRadius;
	};
	
	dim_range zRange;
	dim_range tRange;
	dim_range cRange;
	image_track track;	
};

void
FreeImageLoader(image_loader *loader)
{
	CloseSSIFFile(loader->SSIFfile->SSIFfile);
}

u32
ImageLoaderGetWidth(image_loader *loader)
{
	return(loader->SSIFfile->SSIFfile->header->width);
}

u32
ImageLoaderGetHeight(image_loader *loader)
{
	return(loader->SSIFfile->SSIFfile->header->height);
}

u32
ImageLoaderGetChannels(image_loader *loader)
{
	return(loader->SSIFfile->SSIFfile->header->channels);
}

u32
ImageLoaderGetTimePoints(image_loader *loader)
{
	return(loader->SSIFfile->SSIFfile->header->timepoints);
}

u32
ImageLoaderGetDepth(image_loader *loader)
{
	return(loader->SSIFfile->SSIFfile->header->depth);
}

void
CreateImageLoader(memory_arena *arena, memory_arena *zlibArena, image_loader *loader, program_arguments *pArgs)
{
	char *inputFile = pArgs->inputFile;
	image_track track = pArgs->track;
	dim_range zRange = pArgs->zRange;
	dim_range tRange = pArgs->tRange;
	dim_range cRange = pArgs->cRange;

	loader->SSIFfile = OpenSSIFFileForReading(arena, inputFile);
	loader->channel = (u32)cRange.start;
	loader->timepoint = (u32)tRange.start;
	loader->depth = (u32)zRange.start;
	loader->track = track;
	loader->compressedImageBuffer = PushArrayP(arena, u08, 2 * GetSSIFBytesPerImage(loader->SSIFfile->SSIFfile));
	loader->zlibArena = zlibArena;

	if (pArgs->zRange.end == -1)
	{
		pArgs->zRange.end = (s32)ImageLoaderGetDepth(loader) - 1;
	}
	if (pArgs->tRange.end == -1)
	{
		pArgs->tRange.end = (s32)ImageLoaderGetTimePoints(loader) - 1;
	}
	if (pArgs->cRange.end == -1)
	{
		pArgs->cRange.end = (s32)ImageLoaderGetChannels(loader) - 1;
	}
}

u32 *
GetImageLoaderCurrentTrackIndexAndSize(image_loader *loader, u32 *size)
{
	u32 *index;

	switch (loader->track) 
	{
		case track_channel:
			{
				index = &loader->channel;
				*size = ImageLoaderGetChannels(loader);
			} break;

		case track_timepoint:
			{
				index = &loader->timepoint;
				*size = ImageLoaderGetTimePoints(loader);
			} break;

		case track_depth:
			{
				index = &loader->depth;
				*size = ImageLoaderGetDepth(loader);
			} break;
	}

	return(index);
}

u32
LoadCurrentImageAndAdvanceIndex(image_loader *loader, u08 *imageBuffer)
{
	u32 result = 0;

	u32 trackSize;
	u32 *currentIndex = GetImageLoaderCurrentTrackIndexAndSize(loader, &trackSize);

	if (*currentIndex < trackSize)
	{
		result = 1;

		ReadImageFromSSIFFile(loader->zlibArena, loader->SSIFfile, loader->depth, loader->timepoint, loader->channel, imageBuffer, loader->compressedImageBuffer);

		++(*currentIndex);
	}

	return(result);
}

three_d_volume
GetImageVolume_FromImageLoader(image_loader *loader)
{
	three_d_volume vol;
	CreateThreeDVol_FromInt(&vol, ImageLoaderGetWidth(loader), ImageLoaderGetHeight(loader), 1);

	return(vol);
}

CudaKernel
MemoryCopy_Cuda(u08 *input, u08 *output, u32 bufferLength)
{
	OneDCudaLoop(index, bufferLength)
	{
		*(output + index) = *(input + index);
	}
}

CudaKernel
ConvertFromSixteenBitToEightBit_Cuda(u08 *sixteenBitBuffer, u08 *eightBitBuffer, u32 bufferLength)
{
	OneDCudaLoop(index, bufferLength)
	{
		*(eightBitBuffer + index) = (u08)(((f32)(*(((u16 *)sixteenBitBuffer) + index)) / (f32)u16_max * (f32)u08_max) + 0.5);	
	}
}

CudaKernel
ConvertFromThirtyTwoBitToEightBit_Cuda(u08 *thirtyTwoBitBuffer, u08 *eightBitBuffer, u32 bufferLength)
{
	OneDCudaLoop(index, bufferLength)
	{
		*(eightBitBuffer + index) = (u08)(((f32)(*(((u32 *)thirtyTwoBitBuffer) + index)) / (f32)u32_max * (f32)u08_max) + 0.5);	
	}
}

struct
to_eight_bit_converter
{
	u08 *buffer;
	stream *converterStream;
	u32 bufferSize;
	u32 byteSize;
};

void
CreateToEightBitConverter(to_eight_bit_converter *converter, u32 bufferSize, stream *converterStream, u32 byteSize)
{
	converter->converterStream = converterStream;
	converter->bufferSize = bufferSize;
	converter->byteSize = byteSize;

	CudaMallocManaged(converter->buffer, bufferSize * byteSize);
}

void
FreeToEightBitConverter(to_eight_bit_converter *converter)
{
	cudaFree(converter->buffer);
}

void
ConvertToEightBit(to_eight_bit_converter *converter, u08 *eightBitBuffer)
{
	switch (converter->byteSize)
	{
		case 1:
			{
				LaunchCudaKernel_Simple_Stream(MemoryCopy_Cuda, *converter->converterStream, converter->buffer, eightBitBuffer, converter->bufferSize);
			} break;
		
		case 2:
			{
				LaunchCudaKernel_Simple_Stream(ConvertFromSixteenBitToEightBit_Cuda, *converter->converterStream, converter->buffer, eightBitBuffer, converter->bufferSize);
			} break;

		case 4:
			{
				LaunchCudaKernel_Simple_Stream(ConvertFromThirtyTwoBitToEightBit_Cuda, *converter->converterStream, converter->buffer, eightBitBuffer, converter->bufferSize);
			} break;
	}
	//NOTE: only 8bit, 16bit and 32bit images supported
		
	cudaStreamSynchronize(*converter->converterStream);	
}
#endif
