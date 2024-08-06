# src/fft.jl

#module DaggerFFT
#__precompile__(false)
using Distributed

function closest_factors(n)
    factors = []
    for i in 1:floor(Int, sqrt(n))
        if n % i == 0
            push!(factors, (i, div(n, i)))
        end
    end

    # Sort the factors by the absolute difference between the two factors
    sorted_factors = sort(factors, by = f -> abs(f[1] - f[2]))
    return sorted_factors[1]
end

n = Distributed.nprocs()
factors = closest_factors(n)

using KernelAbstractions, AbstractFFTs, LinearAlgebra, FFTW, Dagger, CUDA, CUDA.CUFFT, Random


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
        if transform isa FFT
            return CUDA.CUFFT.plan_fft(A, dims; kwargs...)
        elseif transform isa IFFT
            return CUDA.CUFFT.plan_ifft(A, dims; kwargs...)
        elseif transform isa FFT!
            return CUDA.CUFFT.plan_fft!(A, dims; kwargs...)
        elseif transform isa IFFT!
            return CUDA.CUFFT.plan_ifft!(A, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
    else
        if transform isa FFT
            return plan_fft(A, dims; kwargs...)
        elseif transform isa IFFT
            return plan_ifft(A, dims; kwargs...)
        elseif transform isa FFT!
            return plan_fft!(A, dims; kwargs...)
        elseif transform isa IFFT!
            return plan_ifft!(A, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
    end
end

function plan_transform(transform, A, dims, n; kwargs...)
    if is_gpu_array(A)
        if transform isa RFFT
            return CUDA.CUFFT.plan_rfft(A, dims; kwargs...)
        elseif transform isa IRFFT
            return CUDA.CUFFT.plan_irfft(A, n, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
    else
        if transform isa RFFT
            return plan_rfft(A, dims; kwargs...)
        elseif transform isa IRFFT
            return plan_irfft(A, n, dims; kwargs...)
        else
            throw(ArgumentError("Unknown transform type"))
        end
    end
end

indexes(a::ArrayDomain) = a.indexes

Base.getindex(arr::CuArray, d::ArrayDomain) = arr[indexes(d)...]

Base.getindex(arr::KernelAbstractions.AbstractArray, d::ArrayDomain) = arr[indexes(d)...]

function Base.similar(x::DArray{T,N}) where {T,N}
    alloc(idx, sz) = KernelAbstractions.AbstractArray{T}(undef, sz)

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

function apply_fft!(out_part, a_part, transform, dim)
    plan = plan_transform(transform, a_part, dim)
    if transform isa Union{FFT!, IFFT!}
        out_part .= plan * a_part  # In-place transform
    else
  #      result = plan * a_part  # Out-of-place transform
  #      copyto!(out_part, result)  # Copy result to out_part
           out_part .= plan * a_part 
    end
end

function apply_fft!(out_part, a_part, transform, dim, n)
    plan = plan_transform(transform, a_part, dim, n)
    out_part .= plan * a_part 
end

function fft(A::AbstractArray{T,3}, transforms, dims) where T
    x, y, z = size(A)
    backend = get_backend(A)

    if T <: Real
        a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
        buffer = DArray(ComplexF64.(A), Blocks(x, div(y, 2), div(z, 2)))
        b = DArray(ComplexF64.(A), Blocks(div(x, 2), y, div(z, 2)))
        c = DArray(ComplexF64.(A), Blocks(div(x, 2), div(y, 2), z))
    else
        a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
        b = DArray(A, Blocks(div(x, 2), y, div(z, 2)))
        c = DArray(A, Blocks(div(x, 2), div(y, 2), z))
    end

    Dagger.spawn_datadeps() do
        if T <: Real
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                buffer_part = buffer.chunks[idx]
                Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
            end

            Dagger.@spawn copyto!(Out(b), In(buffer))
        else
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
            Dagger.@spawn copyto!(Out(b), In(a))
        end

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
        end

        Dagger.@spawn copyto!(Out(c), In(b))
    
        for idx in 1:length(c.chunks)
            c_part = c.chunks[idx]
            Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[3]), In(dims[3]))
        end
    end

    return AbstractCollect(c, backend)
end

function fft(A::AbstractArray{T,2}, transforms, dims) where T
    x, y = size(A)
    backend = get_backend(A)

    if T <: Real
        a = DArray(A, Blocks(x, div(y, 2)))
        buffer = DArray(ComplexF64.(A), Blocks(x, div(y, 2)))
        b = DArray(ComplexF64.(A), Blocks(div(x, 2), y))
    else
        a = DArray(A, Blocks(x, div(y, 2)))
        b = DArray(A, Blocks(div(x, 2), y))
    end

    Dagger.spawn_datadeps() do
        if T <: Real
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                buffer_part = buffer.chunks[idx]
                Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
        else
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
            end
        end

        Dagger.@spawn copyto!(Out(b), In(buffer))

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
        end
    end

    return AbstractCollect(b, backend)
end

function fft(A::AbstractArray{T,1}, transforms, dims) where T
    backend = get_backend(A)

    if T <: Real
        a = DArray(ComplexF64.(A), Blocks(length(A)))
    else
        a = DArray(A, Blocks(length(A)))
    end

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms), In(dims))
        end
    end

    return AbstractCollect(a, backend)
end

function fft(A::AbstractArray, transforms, dims)
    throw(ArgumentError("FFT not implemented for $(length(dims))-dimensional arrays"))
end

function ifft(A::AbstractArray{T,3}, transforms, dims) where T
    x, y, z = size(A)
    backend = get_backend(A)

    a = DArray(A, Blocks(div(x, 2), div(y, 2), z))
    b = DArray(A, Blocks(div(x, 2), y, div(z, 2)))
    c = DArray(A, Blocks(x, div(y, 2), div(z, 2)))

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[3]), In(dims[3]))
        end

        Dagger.@spawn copyto!(Out(b), In(a))

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
        end

        Dagger.@spawn copyto!(Out(c), In(b))

        for idx in 1:length(c.chunks)
            c_part = c.chunks[idx]
            Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[1]), In(dims[1]))
        end
    end

    return AbstractCollect(c, backend)

end


function ifft(A::AbstractArray{T,2}, transforms, dims) where T
    x, y = size(A)
    a = DArray(A, Blocks(div(x, 2), y))
    b = DArray(A, Blocks(x, div(y, 2)))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[2]), In(dims[2]))
        end

        Dagger.@spawn copyto!(Out(b), In(a))

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
        end
    end

    return AbstractCollect(b, backend)
end

function ifft(A::AbstractArray{T,1}, transforms, dims) where T
    a = DArray(A, Blocks(length(A)))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms), In(dims))
        end
    end

    return AbstractCollect(a, backend)
end

function ifft(A::AbstractArray, transforms, dims)
    throw(ArgumentError("IFFT not implemented for $(length(dims))-dimensional arrays"))
end


function rfft(A::AbstractArray{T,3}, transforms, dims) where T
    x, y, z = size(A)
    a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
    buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(x, div(y, 2), div(z, 2)))
    b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2), y, div(z, 2)))
    c = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2), div(y, 2), z))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            buffer_part = buffer.chunks[idx]
            Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(z))
        end
        Dagger.@spawn copyto!(Out(b), In(buffer))

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
        end

        Dagger.@spawn copyto!(Out(c), In(b))

        for idx in 1:length(c.chunks)
            c_part = c.chunks[idx]
            Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[3]), In(dims[3]))
        end
    end

    return AbstractCollect(c, backend)
end

function rfft(A::AbstractArray{T,2}, transforms, dims) where T
    x, y = size(A)
    a = DArray(A, Blocks(x, div(y, 2)))
    buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(x, div(y, 2)))
    b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(div(x, 2), y))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            buffer_part = buffer.chunks[idx]
            Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(y))
        end

        Dagger.@spawn copyto!(Out(b), In(buffer))

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
        end
    end

    return AbstractCollect(b, backend)
end


function rfft(A::AbstractArray{T,1}, transforms, dims) where T
    a = DArray(A, Blocks(length(A)))
    x = length(A)
    buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1]), Blocks(length(A)))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            buffer_part = buffer.chunks[idx]
            Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms), In(dims), In(x))
        end
    end
    
    return AbstractCollect(buffer, backend)
end

function rfft(A::AbstractArray, transforms, dims)
    throw(ArgumentError("FFT not implemented for $(length(dims))-dimensional arrays"))
end


function irfft(A::AbstractArray{T,3}, transforms, dims) where T
    x, y, z = size(A)
    a = DArray(A, Blocks(div(x, 2), div(y, 2), z))
    b = DArray(A, Blocks(div(x, 2), y, div(z, 2)))
    c = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
    buffer = DArray(similar(A, Float64, ((x - 1) * 2, y, z)), Blocks(((x - 1) * 2), div(y, 2), div(z, 2)))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[3]), In(dims[3]))
        end
        Dagger.@spawn copyto!(Out(b), In(a))

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
        end
        Dagger.@spawn copyto!(Out(c), In(b))

        for idx in 1:length(c.chunks)
            c_part = c.chunks[idx]
            buffer_part = buffer.chunks[idx]
            Dagger.@spawn apply_fft!(Out(buffer_part), In(c_part), In(transforms[1]), In(dims[1]), In(z))
        end
    end

    return AbstractCollect(buffer, backend)

end

function irfft(A::AbstractArray{T,2}, transforms, dims) where T
    x, y = size(A)
    a = DArray(A, Blocks(x, div(y, 2)))
    b = DArray(A, Blocks(div(x, 2) + 1, y))
    buffer = DArray(similar(A, Float64, ((x - 1) * 2, y)), Blocks(((x - 1) * 2), div(y, 2)))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[2]), In(dims[2]))
        end

        Dagger.@spawn copyto!(Out(b), In(a))

        for idx in 1:length(b.chunks)
            b_part = b.chunks[idx]
            buffer_part = buffer.chunks[idx]
            Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), In(transforms[1]), In(dims[1]), In(y))
        end
    end

    return AbstractCollect(buffer, backend)
end

function irfft(A::AbstractArray{T,1}, transforms, dims) where T
    x = length(A)
    a = DArray(A, Blocks(length(A)))
    buffer = DArray(similar(A, Float64, ((x - 1) * 2)), Blocks(((x - 1) * 2)))
    backend = get_backend(A)

    Dagger.spawn_datadeps() do
        for idx in 1:length(a.chunks)
            a_part = a.chunks[idx]
            buffer_part = buffer.chunks[idx]
            Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms), In(dims), In(x))
        end
    end

    return AbstractCollect(buffer, backend)
end

function irfft(A::AbstractArray, transforms, dims)
    throw(ArgumentError("IFFT not implemented for $(length(dims))-dimensional arrays"))
end

#end
