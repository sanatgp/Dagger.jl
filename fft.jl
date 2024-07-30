# src/fft.jl

#module DaggerFFT
#__precompile__(false)


using AbstractFFTs
using LinearAlgebra
using FFTW
using Dagger: DArray, @spawn, InOut, In
using Dagger
using CUDA, CUDA.CUFFT
using Random
using KernelAbstractions


struct FFT end
struct RFFT end
struct IRFFT end
struct IFFT end
struct FFT! end
struct RFFT! end
struct IRFFT! end
struct IFFT! end


export FFT, RFFT, IRFFT, IFFT, FFT!, RFFT!, IRFFT!, IFFT!, fft, ifft

function plan_transform(transform, A, dims; kwargs...)
    if is_gpu_array(A)
        if transform isa RFFT
            return CUDA.CUFFT.plan_rfft(A, dims; kwargs...)
        elseif transform isa FFT
            return CUDA.CUFFT.plan_fft(A, dims; kwargs...)
        elseif transform isa IRFFT
            return CUDA.CUFFT.plan_irfft(A, dims; kwargs...)
        elseif transform isa IFFT
            return CUDA.CUFFT.plan_ifft(A, dims; kwargs...)
        elseif transform isa RFFT!
            return CUDA.CUFFT.plan_rfft!(A, dims; kwargs...)
        elseif transform isa FFT!
            return CUDA.CUFFT.plan_fft!(A, dims; kwargs...)
        elseif transform isa IRFFT!
            return CUDA.CUFFT.plan_irfft!(A, dims; kwargs...)
        elseif transform isa IFFT!
            return CUDA.CUFFT.plan_ifft!(A, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
    else
        if transform isa RFFT
            return plan_rfft(A, dims; kwargs...)
        elseif transform isa FFT
            return plan_fft(A, dims; kwargs...)
        elseif transform isa IRFFT
            return plan_irfft(A, dims; kwargs...)
        elseif transform isa IFFT
            return plan_ifft(A, dims; kwargs...)
        elseif transform isa RFFT!
            return plan_rfft!(A, dims; kwargs...)
        elseif transform isa FFT!
            return plan_fft!(A, dims; kwargs...)
        elseif transform isa IRFFT!
            return plan_irfft!(A, dims; kwargs...)
        elseif transform isa IFFT!
            return plan_ifft!(A, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
    end
end

indexes(a::ArrayDomain) = a.indexes

Base.getindex(arr::CuArray, d::ArrayDomain) = arr[indexes(d)...]

Base.getindex(arr::KernelAbstractions.AbstractArray, d::ArrayDomain) = arr[indexes(d)...]

function Base.similar(x::DArray{T,N}) where {T,N}
    alloc(idx, sz) = CuArray{T,N}(undef, sz)
    thunks = [Dagger.@spawn alloc(i, size(x)) for (i, x) in enumerate(x.subdomains)]
    return DArray(T, x.domain, x.subdomains, thunks, x.partitioning, x.concat)
end

function AbstractCollect(d::DArray{T,N}, backend=KernelAbstractions.CPU()) where {T,N}
    total_size = size(d)
    result = KernelAbstractions.zeros(backend, T, total_size...)
    
    for (idx, chunk) in enumerate(d.chunks)
        chunk_domain = d.subdomains[idx]
        fetched_chunk = fetch(chunk)
        
        # Copy the chunk data to the appropriate part of the result array
        indices = map(r -> r.start:r.stop, chunk_domain.indexes)
        result[indices...] = fetched_chunk
    end
    
    return result
end

function is_gpu_array(A)
    return A isa CuArray
end

function apply_fft(a_part, transform, dim)
    plan = plan_transform(transform, a_part, dim)
    if transform isa Union{FFT!, RFFT!, IRFFT!, IFFT!}
        return plan * a_part  # In-place transform
    else
        return plan * a_part  # Out-of-place transform
    end
end

function fft(A::AbstractArray{T,3}, transforms, dims) where T
        x, y, z = size(A)
        a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
        a_parts = a.chunks
        b = DArray(similar(A), Blocks(div(x, 2), y, div(z, 2)))
        b_parts = b.chunks
        c = DArray(similar(A), Blocks(div(x, 2), div(y, 2), z))
        c_parts = c.chunks
        backend = get_backend(A)

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a.chunks[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(transforms[1]), In(dims[1]))
            end
            Dagger.@spawn copyto!(Out(b), In(a))
        end

        Dagger.spawn_datadeps() do
            for idx in 1:length(b_parts)
                b_part = b.chunks[idx]
                b.chunks[idx] = Dagger.@spawn apply_fft(InOut(b_part), In(transforms[2]), In(dims[2]))
            end
            Dagger.@spawn copyto!(Out(c), In(b))
        end
        
        Dagger.spawn_datadeps() do
            for idx in 1:length(c_parts)
                c_part = c.chunks[idx]
                c.chunks[idx] = Dagger.@spawn apply_fft(InOut(c_part), In(transforms[3]), In(dims[3]))
            end
        end

    return AbstractCollect(c, backend)
end

function fft(A::AbstractArray{T,2}, transforms, dims) where T
    x, y = size(A)
    a = DArray(A, Blocks(x, div(y, 2)))
    a_parts = a.chunks
    b = DArray(similar(A), Blocks(div(x, 2), y))
    b_parts = b.chunks

    Dagger.spawn_datadeps() do
        for idx in 1:length(a_parts)
            a_part = a_parts[idx]
            a.chunks[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(transforms[1]), In(dims[1]))
        end
        Dagger.@spawn copyto!(Out(b), In(a))
    end

    Dagger.spawn_datadeps() do
        for idx in 1:length(b_parts)
            b_part = b.chunks[idx]
            b.chunks[idx] = Dagger.@spawn apply_fft(InOut(b_part), In(transforms[2]), In(dims[2]))
        end
    end
    
    backend = get_backend(A)
    return AbstractCollect(b, backend)
end

function fft(A::AbstractArray{T,1}, transforms, dims) where T
    a = DArray(A, Blocks(length(A)))
    a_parts = a.chunks

    Dagger.spawn_datadeps() do
        for idx in 1:length(a_parts)
            a_part = a_parts[idx]
            a.chunks[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(transforms[1]), In(dims[1]))
        end
    end
    
    backend = get_backend(A)
    return AbstractCollect(a, backend)
end

function fft(A::AbstractArray, transforms, dims)
    throw(ArgumentError("FFT not implemented for $(length(dims))-dimensional arrays"))
end


function ifft(A::AbstractArray{T,3}, transforms, dims) where T
        x, y, z = size(A)
        a = DArray(A, Blocks(div(x, 2), div(y, 2), z))
        a_parts = a.chunks
        b = DArray(similar(A), Blocks(div(x, 2), y, div(z, 2)))
        b_parts = b.chunks
        c = DArray(similar(A), Blocks(x, div(y, 2), div(z, 2)))
        c_parts = c.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a.chunks[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(transforms[3]), In(dims[3]))
            end
            Dagger.@spawn copyto!(Out(b), In(a))
        end

        Dagger.spawn_datadeps() do
            for idx in 1:length(b_parts)
                b_part = b.chunks[idx]
                b.chunks[idx] = Dagger.@spawn apply_fft(InOut(b_part), In(transforms[2]), In(dims[2]))
            end
            Dagger.@spawn copyto!(Out(c), In(b))
        end
        
        Dagger.spawn_datadeps() do
            for idx in 1:length(c_parts)
                c_part = c.chunks[idx]
                c.chunks[idx] = Dagger.@spawn apply_fft(InOut(c_part), In(transforms[1]), In(dims[1]))
            end
        end
    
        backend = get_backend(A)
    return AbstractCollect(c, backend)
end

function ifft(A::AbstractArray{T,2}, transforms, dims) where T
    x, y, z = size(A)
    a = DArray(A, Blocks(div(x, 2), div(y, 2)))
    a_parts = a.chunks
    b = DArray(similar(A), Blocks(div(x, 2), y))
    b_parts = b.chunks

    Dagger.spawn_datadeps() do
        for idx in 1:length(a_parts)
            a_part = a_parts[idx]
            a.chunks[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(transforms[3]), In(dims[3]))
        end
        Dagger.@spawn copyto!(Out(b), In(a))
    end

    Dagger.spawn_datadeps() do
        for idx in 1:length(b_parts)
            b_part = b.chunks[idx]
            b.chunks[idx] = Dagger.@spawn apply_fft(InOut(b_part), In(transforms[2]), In(dims[2]))
        end
    end

    backend = get_backend(A)
    return AbstractCollect(b, backend)
end

function ifft(A::AbstractArray{T,1}, transforms, dims) where T
    x, y, z = size(A)
    a = DArray(A, Blocks(div(x, 2), div(y, 2), z))
    a_parts = a.chunks

    Dagger.spawn_datadeps() do
        for idx in 1:length(a_parts)
            a_part = a_parts[idx]
            a.chunks[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(transforms[3]), In(dims[3]))
        end
    end

    backend = get_backend(A)
    return AbstractCollect(a, backend)
end

function ifft(A::AbstractArray, transforms, dims)
    throw(ArgumentError("IFFT not implemented for $(length(dims))-dimensional arrays"))
end

#end
