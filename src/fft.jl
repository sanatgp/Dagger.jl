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

using KernelAbstractions, AbstractFFTs, LinearAlgebra, FFTW, Dagger, CUDA, CUDA.CUFFT, Random, GPUArrays, AMDGPU


const R2R_SUPPORTED_KINDS = (
    FFTW.DHT,
    FFTW.REDFT00,
    FFTW.REDFT01,
    FFTW.REDFT10,
    FFTW.REDFT11,
    FFTW.RODFT00,
    FFTW.RODFT01,
    FFTW.RODFT10,
    FFTW.RODFT11,
)

struct FFT end
struct RFFT end
struct IRFFT end
struct IFFT end
struct FFT! end
struct RFFT! end
struct IRFFT! end
struct IFFT! end


export FFT, RFFT, IRFFT, IFFT, FFT!, RFFT!, IRFFT!, IFFT!, fft, ifft, R2R, R2R!

struct R2R{K}
    kind::K
    function R2R(kind::K) where {K}
        if kind ∉ R2R_SUPPORTED_KINDS
            throw(ArgumentError("Unsupported R2R transform kind: $(kind2string(kind))"))
        end
        new{K}(kind)
    end
end

struct R2R!{K}
    kind::K
    function R2R!(kind::K) where {K}
        if kind ∉ R2R_SUPPORTED_KINDS
            throw(ArgumentError("Unsupported R2R transform kind: $(kind2string(kind))"))
        end
        new{K}(kind)
    end
end

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
        elseif transform isa R2R
            return plan_r2r(A, dims, kind(transform); kwargs...)
        elseif transform isa R2R!
            return plan_r2r!(A, dims, kind(transform); kwargs...)
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

kind(transform::R2R) = transform.kind
kind(transform::R2R!) = transform.kind

function plan_transform(transform::Union{R2R, R2R!}, A, dims; kwargs...)
    kd = kind(transform)
    if transform isa R2R
        return FFTW.plan_r2r(A, kd, dims; kwargs...)
    elseif transform isa R2R!
        return FFTW.plan_r2r!(A, kd, dims; kwargs...)
    else
        throw(ArgumentError("Unknown transform type"))
    end
end

indexes(a::ArrayDomain) = a.indexes

Base.getindex(arr::CuArray, d::ArrayDomain) = arr[indexes(d)...]

Base.getindex(arr::KernelAbstractions.AbstractArray, d::ArrayDomain) = arr[indexes(d)...]

Base.getindex(arr::GPUArrays.AbstractGPUArray, d::ArrayDomain) = arr[indexes(d)...]

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

function create_darray(A::AbstractArray{T,N}, blocks::Blocks{N}) where {T,N} #TODO:Creates view?
    domain = ArrayDomain(map(Base.OneTo, size(A)))
    
    #calculate subdomain
    dims = size(A)
    block_sizes = blocks.blocksize
    subdomain_sizes = map((d, b) -> [b for _ in 1:ceil(Int, d/b)], dims, block_sizes)
    subdomain_cumlengths = map(cumsum, subdomain_sizes)
    
    #create subdomains
    subdomains = Array{ArrayDomain{N}, N}(undef, map(length, subdomain_sizes))
    for idx in CartesianIndices(subdomains)
        starts = map((cumlength, i) -> i == 1 ? 1 : cumlength[i-1] + 1, subdomain_cumlengths, idx.I)
        ends = map(getindex, subdomain_cumlengths, idx.I)
        subdomains[idx] = ArrayDomain(map((s, e) -> s:e, starts, ends))
    end
    
    #create chunks
    chunks = Array{Any,N}(undef, size(subdomains))
    for idx in CartesianIndices(chunks)
        subdomain = subdomains[idx]
        view_indices = subdomain.indexes
        chunks[idx] = Dagger.tochunk(view(A, view_indices...))
    end
    
    DArray{T,N,typeof(blocks),typeof(cat)}(
        domain,
        subdomains,
        chunks,
        blocks,
        cat
    )
end

function is_gpu_array(A)
    return A isa Union{CuArray, ROCArray}
end

function apply_fft!(out_part, a_part, transform, dim)
    plan = plan_transform(transform, a_part, dim)
    if transform isa Union{FFT!, IFFT!}
        out_part .= plan * a_part  # In-place transform
    else
  #      result = plan * a_part  # Out-of-place transform
  #      copyto!(out_part, result)
           out_part .= plan * a_part 
    end
end

function apply_fft!(out_part, a_part, transform, dim, n)
    plan = plan_transform(transform, a_part, dim, n)
    out_part .= plan * a_part 
end
"""
function transpose_pencils(src::DArray, dst::DArray)
    for (src_idx, src_chunk) in enumerate(src.chunks)
        src_data = fetch(src_chunk)
        for (dst_idx, dst_chunk) in enumerate(dst.chunks)
            dst_domain = dst.subdomains[dst_idx]
            src_domain = src.subdomains[src_idx]
            
            #Calculate the intersection of source and destination domains
            intersect_domain = intersect(src_domain, dst_domain)
            
            if !isempty(intersect_domain)
                #calculate relative indices
                src_indices = relative_indices(intersect_domain, src_domain)
                dst_indices = relative_indices(intersect_domain, dst_domain)
                
                dst_chunk_data = fetch(dst_chunk)
                
                if size(dst_chunk_data) != size(dst_domain)
                    dst_chunk_data = similar(dst_chunk_data, size(dst_domain)...)
                end
                
                #Scatter
                dst_chunk_data[dst_indices...] = src_data[src_indices...]
                dst.chunks[dst_idx] = Dagger.tochunk(dst_chunk_data)
            end
        end
    end
end
"""

@kernel function transpose_kernel!(dst, src, src_size_x, src_size_y, src_size_z,
                            dst_size_x, dst_size_y, dst_size_z,
                            src_offset_x, src_offset_y, src_offset_z,
                            dst_offset_x, dst_offset_y, dst_offset_z)
    i, j, k = @index(Global, NTuple)

    src_i = i + src_offset_x
    src_j = j + src_offset_y
    src_k = k + src_offset_z

    dst_i = i + dst_offset_x
    dst_j = j + dst_offset_y
    dst_k = k + dst_offset_z

    if src_i <= src_size_x && src_j <= src_size_y && src_k <= src_size_z &&
        dst_i <= dst_size_x && dst_j <= dst_size_y && dst_k <= dst_size_z
        dst[dst_i, dst_j, dst_k] = src[src_i, src_j, src_k]
    end
end

function transpose_pencils(src::DArray, dst::DArray)
    for (src_idx, src_chunk) in enumerate(src.chunks)
        src_data = fetch(src_chunk)
        for (dst_idx, dst_chunk) in enumerate(dst.chunks)
            dst_domain = dst.subdomains[dst_idx]
            src_domain = src.subdomains[src_idx]

            intersect_domain = intersect(src_domain, dst_domain)

            if !isempty(intersect_domain)

                src_indices = relative_indices(intersect_domain, src_domain)
                dst_indices = relative_indices(intersect_domain, dst_domain)

                dst_chunk_data = fetch(dst_chunk)

                if size(dst_chunk_data) != size(dst_domain)
                  dst_chunk_data = similar(dst_chunk_data, size(dst_domain)...)
                end

                intersect_size = map(r -> length(r), intersect_domain.indexes)

                #offsets
                src_offset = map(r -> r.start - 1, src_indices)
                dst_offset = map(r -> r.start - 1, dst_indices)

                backend = get_backend(src_data)

                kernel = transpose_kernel!(backend)
                kernel(dst_chunk_data, src_data, 
                  size(src_data)..., size(dst_chunk_data)...,
                  src_offset..., dst_offset..., ndrange=intersect_size)
                KernelAbstractions.synchronize(backend)

                dst.chunks[dst_idx] = Dagger.tochunk(dst_chunk_data)
            end
        end
    end         
end

function relative_indices(sub_domain, full_domain)
    return map(pair -> (pair[1].start:pair[1].stop) .- (pair[2].start - 1), 
               zip(sub_domain.indexes, full_domain.indexes))
end

"
user should call the function with:
    Dagger.fft(A, transforms, dims) #default pencil
    Dagger.fft(A, transforms, dims, decomp=:pencil)
    Dagger.fft(A, transforms, dims, decomp=:slab)
    transforms = (FFT(), FFT(), FFT())
or    transforms = [R2R(FFTW.REDFT10), R2R(FFTW.REDFT10), R2R(FFTW.REDFT10)]
or    transforms = (RFFT(), FFT(), FFT())
    dims = (1, 2, 3)
"
function fft(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{FFT,IFFT,RFFT,IRFFT,R2R,FFT!,IFFT!,R2R!}},
    dims::NTuple{N,Int};
    decomp::Union{Symbol,Nothing}=nothing
) where {T,N}
    # default to pencil decomposition
    decomp = isnothing(decomp) ? :pencil : decomp

    if decomp ∉ [:pencil, :slab]
        throw(ArgumentError("Invalid decomposition type. Use :pencil or :slab"))
    end

    if N == 3
        x, y, z = size(A)
        backend = get_backend(A)

        if transforms[1] isa RFFT
            a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
            buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2) + 1, div(y, 2), div(z, 2)))
            b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), decomp == :pencil ? Blocks(div(x, 2) + 1, y, div(z, 2)) : Blocks(div(x, 2) + 1, y, div(z, 2)))
            c = DArray(ComplexF64.(A[1:div(x, 2) + 1, :, :]), Blocks(div(x, 2) + 1, div(y, 2), z))
        elseif T <: Real && all(transform -> transform isa FFT, transforms)
            a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
            buffer = DArray(ComplexF64.(A), Blocks(x, div(y, 2), div(z, 2)))
            b = DArray(ComplexF64.(A), decomp == :pencil ? Blocks(div(x, 2), y, div(z, 2)) : Blocks(x, y, div(z, 2)))
            c = DArray(ComplexF64.(A), Blocks(div(x, 2), div(y, 2), z))
        else
            a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
            b = DArray(A, decomp == :pencil ? Blocks(div(x, 2), y, div(z, 2)) : Blocks(x, y, div(z, 2)))
            c = DArray(A, Blocks(div(x, 2), div(y, 2), z))
        end

        Dagger.spawn_datadeps() do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(z))
                end
                Dagger.@spawn transpose_pencils(In(buffer), Out(b))
            elseif T <: Real && all(transform -> transform isa FFT, transforms)
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose_pencils(In(buffer), Out(b))
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                if decomp == :pencil
                    Dagger.@spawn transpose_pencils(In(a), Out(b))
                else
                    b = a  # For slabs, skip the first transpose
                end
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end

            Dagger.@spawn transpose_pencils(In(b), Out(c))
        
            for idx in 1:length(c.chunks)
                c_part = c.chunks[idx]
                Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[3]), In(dims[3]))
            end
        end

        return AbstractCollect(c, backend)

    elseif N == 2
        x, y = size(A)
        backend = get_backend(A)

        if transforms[1] isa RFFT
            a = DArray(A, Blocks(x, div(y, 2)))
            buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(div(x, 2) + 1, div(y, 2)))
            b = DArray(ComplexF64.(A[1:div(x, 2) + 1, :]), Blocks(div(x, 2) + 1, y))
        elseif T <: Real
            a = DArray(A, decomp == :pencil ? Blocks(x, div(y, 2)) : Blocks(x, y))
            buffer = DArray(ComplexF64.(A), decomp == :pencil ? Blocks(x, div(y, 2)) : Blocks(x, y))
            b = DArray(ComplexF64.(A), Blocks(div(x, 2), y))
        else
            a = DArray(A, decomp == :pencil ? Blocks(x, div(y, 2)) : Blocks(x, y))
            b = DArray(A, Blocks(div(x, 2), y))
        end

        Dagger.spawn_datadeps() do
            if transforms[1] isa RFFT
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(y))
                end
                Dagger.@spawn transpose_pencils(In(buffer), Out(b))
            elseif T <: Real
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                Dagger.@spawn transpose_pencils(In(buffer), Out(b))
            else
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[1]), In(dims[1]))
                end
                if decomp == :pencil
                    Dagger.@spawn transpose_pencils(In(a), Out(b))
                else
                    b = a  # For slabs, skip the transpose
                end
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end
        end

        return AbstractCollect(b, backend)

    elseif N == 1
        backend = get_backend(A)

        if transforms[1] isa RFFT
            a = DArray(A, Blocks(length(A)))
            x = length(A)
            buffer = DArray(ComplexF64.(A[1:div(x, 2) + 1]), Blocks(div(x, 2) + 1))
            
            Dagger.spawn_datadeps() do
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(x))
                end
            end
            
            return AbstractCollect(buffer, backend)
        else
            if T <: Real
                a = DArray(ComplexF64.(A), Blocks(length(A)))
            else
                a = DArray(A, Blocks(length(A)))
            end

            Dagger.spawn_datadeps() do
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), transforms[1], In(dims[1]))
                end
            end

            return AbstractCollect(a, backend)
        end

    else
        throw(ArgumentError("FFT not implemented for $(N)-dimensional arrays"))
    end
end
function ifft(
    A::AbstractArray{T,N},
    transforms::NTuple{N,Union{FFT,IFFT,RFFT,IRFFT,R2R,FFT!,IFFT!,R2R!}},
    dims::NTuple{N,Int};
    decomp::Union{Symbol,Nothing}=nothing
) where {T,N}
    # default pencil decomposition
    decomp = isnothing(decomp) ? :pencil : decomp

    if decomp ∉ [:pencil, :slab]
        throw(ArgumentError("Invalid decomposition type. Use :pencil or :slab"))
    end

    if N == 3
        x, y, z = size(A)
        backend = get_backend(A)

        if transforms[1] isa IRFFT
            a = DArray(A, Blocks(div(x, 2), div(y, 2), z))
            b = DArray(A, Blocks(div(x, 2), y, div(z, 2)))
            c = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y, z)), Blocks(((x - 1) * 2), div(y, 2), div(z, 2)))
        else
            if decomp == :pencil
                a = DArray(A, Blocks(div(x, 2), div(y, 2), z))
                b = DArray(A, Blocks(div(x, 2), y, div(z, 2)))
                c = DArray(A, Blocks(x, div(y, 2), div(z, 2)))
            else # :slab
                a = DArray(A, Blocks(x, y, div(z, 2)))
                b = DArray(A, Blocks(x, y, div(z, 2)))
                c = DArray(A, Blocks(x, y, div(z, 2)))
            end
        end

        Dagger.spawn_datadeps() do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[3]), In(dims[3]))
            end

            if decomp == :pencil
                Dagger.@spawn transpose_pencils(In(a), Out(b))
            else
                b = a 
            end

            for idx in 1:length(b.chunks)
                b_part = b.chunks[idx]
                Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[2]), In(dims[2]))
            end

            Dagger.@spawn transpose_pencils(In(b), Out(c))

            if transforms[1] isa IRFFT
                for idx in 1:length(c.chunks)
                    c_part = c.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(c_part), In(transforms[1]), In(dims[1]), In(z))
                end
                return AbstractCollect(buffer, backend)
            else
                for idx in 1:length(c.chunks)
                    c_part = c.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(c_part), In(c_part), In(transforms[1]), In(dims[1]))
                end
                return AbstractCollect(c, backend)
            end
        end

    elseif N == 2
        x, y = size(A)
        backend = get_backend(A)

        if transforms[1] isa IRFFT
            a = DArray(A, Blocks(div(x, 2), y))
            b = DArray(A, Blocks(x, div(y, 2)))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2, y)), Blocks(((x - 1) * 2), div(y, 2)))
        else
            if decomp == :pencil
                a = DArray(A, Blocks(div(x, 2), y))
                b = DArray(A, Blocks(x, div(y, 2)))
            else # :slab
                a = DArray(A, Blocks(x, div(y, 2)))
                b = DArray(A, Blocks(x, div(y, 2)))
            end
        end

        Dagger.spawn_datadeps() do
            for idx in 1:length(a.chunks)
                a_part = a.chunks[idx]
                Dagger.@spawn apply_fft!(Out(a_part), In(a_part), In(transforms[2]), In(dims[2]))
            end

            if decomp == :pencil
                Dagger.@spawn transpose_pencils(In(a), Out(b))
            else
                b = a 
            end

            if transforms[1] isa IRFFT
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(b_part), In(transforms[1]), In(dims[1]), In(y))
                end
                return AbstractCollect(buffer, backend)
            else
                for idx in 1:length(b.chunks)
                    b_part = b.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(b_part), In(b_part), In(transforms[1]), In(dims[1]))
                end
                return AbstractCollect(b, backend)
            end
        end

    elseif N == 1
        backend = get_backend(A)
        x = length(A)

        if transforms[1] isa IRFFT
            a = DArray(A, Blocks(length(A)))
            buffer = DArray(similar(A, Float64, ((x - 1) * 2)), Blocks(((x - 1) * 2)))

            Dagger.spawn_datadeps() do
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    buffer_part = buffer.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(buffer_part), In(a_part), In(transforms[1]), In(dims[1]), In(x))
                end
            end

            return AbstractCollect(buffer, backend)
        else
            a = DArray(A, Blocks(length(A)))

            Dagger.spawn_datadeps() do
                for idx in 1:length(a.chunks)
                    a_part = a.chunks[idx]
                    Dagger.@spawn apply_fft!(Out(a_part), In(a_part), transforms[1], In(dims[1]))
                end
            end

            return AbstractCollect(a, backend)
        end

    else
        throw(ArgumentError("IFFT not implemented for $(N)-dimensional arrays"))
    end
end
