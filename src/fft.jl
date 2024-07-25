# src/fft.jl

#module DaggerFFT

using AbstractFFTs
using LinearAlgebra
using FFTW
using Dagger: DArray, @spawn, InOut, In
using Dagger

struct FFT end
struct RFFT end
struct IRFFT end
struct IFFT end
struct FFT! end
struct RFFT! end
struct IRFFT! end
struct IFFT! end

export FFT, RFFT, IRFFT, IFFT, FFT!, RFFT!, IRFFT!, IFFT!, fft, ifft

# Plan transform function
function plan_transform(transform, A, dims; kwargs...)
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

# Apply FFT function
function apply_fft(a_part, transform, dim)
    plan = plan_transform(transform, a_part, dim)
    if transform isa Union{FFT!, RFFT!, IRFFT!, IFFT!}
        plan * a_part  # In-place transform
    else
        plan * a_part  # Out-of-place transform
    end
end

function fft(A, transforms, dims)
    if length(dims) == 3
        x, y, z = size(A)
        a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))

        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(In(a_part), In(transforms[1]), In(dims[1]))
            end
        end
        
        a = DArray(a, Blocks(div(x, 2), y, div(z, 2)))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(In(a_part), In(transforms[2]), In(dims[2]))
            end
        end
        
        a = DArray(a, Blocks(div(x, 2), div(y, 2), z))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(In(a_part), In(transforms[3]), In(dims[3]))
            end
        end
    elseif length(dims) == 2
        x, y = size(A)
        a = DArray(A, Blocks(x, div(y, 2)))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(transforms[1]), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end
        
        a = DArray(a, Blocks(div(x, 2), y))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(transforms[2]), InOut(a_part), In(dims[2]))
                    plan * a_part
                end
            end
        end

    elseif length(dims) == 1
        x = size(A)
        a = DArray(A, Blocks(x))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(transforms[1]), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end

    else
        throw(ArgumentError("Invalid number of dimensions"))
    end

    return collect(a)
end

function ifft(A, transforms, dims)
    if length(dims) == 3
        x, y, z = size(A)
        a = DArray(A, Blocks(div(x, 2), div(y, 2), z))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(In(a_part), In(IFFT()), In(dims[3]))
            end
        end

        a = DArray(a, Blocks(div(x, 2), y, div(z, 2)))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(In(a_part), In(IFFT!()), In(dims[2]))
            end
        end
        
        a = DArray(a, Blocks(x, div(y, 2), div(z, 2)))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(In(a_part), In(IFFT()), In(dims[1]))
            end
        end

    elseif length(dims) == 2
        x, y = size(A)
        a = DArray(A, Blocks(div(x, 2), y))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(IFFT()), InOut(a_part), In(dims[2]))
                    plan * a_part
                end
            end
        end

        a = DArray(a, Blocks(x, div(y, 2)))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(IFFT()), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end

    elseif length(dims) == 1
        x = size(A)
        a = DArray(A, Blocks(x))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(IFFT()), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end

    else
        throw(ArgumentError("Invalid number of dimensions"))
    end
    return collect(a)
end


function rfft(A, dims)
    if length(dims) == 3
        x, y, z = size(A)
        z_r = div(z, 2) + 1

        a = DArray(A, Blocks(x, div(y, 2), div(z, 2)))

        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:3
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(RFFT()), In(dims[1]))
            end
        end

        a = DArray(a, Blocks(div(x, 2), y, div(z, 2)))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(RFFT()), In(dims[2]))
            end
        end

        a = DArray(a, Blocks(div(x, 2), div(y, 2), z_r))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(RFFT()), In(dims[3]))
            end
        end
    elseif length(dims) == 2
        x, y = size(A)
        y_r = div(y, 2) + 1

        a = DArray(A, Blocks(x, y_r))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(RFFT()), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end

        a = DArray(a, Blocks(div(x, 2), y_r))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(RFFT()), InOut(a_part), In(dims[2]))
                    plan * a_part
                end
            end
        end
    elseif length(dims) == 1
        x = size(A, 1)
        x_r = div(x, 2) + 1

        a = DArray(A, Blocks(x_r))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(RFFT()), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end
    else
        throw(ArgumentError("Invalid number of dimensions"))
    end

    return collect(a)
end

function irfft(A, dims, orig_dims)
    if length(dims) == 3
        x, y, z_r = size(A)
        z = orig_dims[3]

        a = DArray(A, Blocks(div(x, 2), div(y, 2), z_r))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(IRFFT()), In(dims[3]))
            end
        end

        a = DArray(a, Blocks(div(x, 2), y, z))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(IRFFT()), In(dims[2]))
            end
        end

        a = DArray(a, Blocks(x, div(y, 2), z))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn apply_fft(InOut(a_part), In(IRFFT()), In(dims[1]))
            end
        end
    elseif length(dims) == 2
        x, y_r = size(A)
        y = orig_dims[2]

        a = DArray(A, Blocks(div(x, 2), y_r))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(IRFFT()), InOut(a_part), In(dims[2]))
                    plan * a_part
                end
            end
        end

        a = DArray(a, Blocks(x, y))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(IRFFT()), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end
    elseif length(dims) == 1
        x_r = size(A, 1)
        x = orig_dims[1]

        a = DArray(A, Blocks(x_r))
        a_parts = a.chunks

        Dagger.spawn_datadeps() do
            for idx in 1:length(a_parts)
                a_part = a_parts[idx]
                a_parts[idx] = Dagger.@spawn begin
                    plan = plan_transform(In(IRFFT()), InOut(a_part), In(dims[1]))
                    plan * a_part
                end
            end
        end
    else
        throw(ArgumentError("Invalid number of dimensions"))
    end

    return collect(a)
end

#end