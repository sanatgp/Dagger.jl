using BenchmarkTools
using FFTW
using Random
using Distributed
using Dagger
using CUDA
using Statistics
using Printf
include("../src/fft.jl")

const SUITE = BenchmarkGroup()

function format_bytes(bytes)
    if bytes < 1024
        return @sprintf("%.2f B", bytes)
    elseif bytes < 1024^2
        return @sprintf("%.2f KB", bytes/1024)
    else
        return @sprintf("%.2f MB", bytes/1024^2)
    end
end

function format_timing(time_ns)
    if time_ns < 1000
        return @sprintf("%.2f ns", time_ns)
    elseif time_ns < 1_000_000
        return @sprintf("%.2f μs", time_ns/1000)
    else
        return @sprintf("%.2f ms", time_ns/1_000_000)
    end
end

function print_size_info(input)
    bytes = sizeof(input)
    elements = length(input)
    dims = size(input)
    println("Input size: $(dims)")
    println("Number of elements: $(elements)")
    println("Memory footprint: $(format_bytes(bytes))")
end

function run_benchmark(input, transform_type, dims; description="")
    print_size_info(input)
    result = @benchmark fft($input, $transform_type, $dims)
    
    println("$description Timing:")
    println("  Min:    $(format_timing(minimum(result.times)))")
    println("  Median: $(format_timing(median(result.times)))")
    println("  Mean:   $(format_timing(mean(result.times)))")
    println("  Max:    $(format_timing(maximum(result.times)))")
    println("  Memory: $(format_bytes(result.memory))")
    println("-" ^ 50) 
    return result
end

# Test different sizes
sizes_2d = [(32,32), (64,64), (128,128), (256,256)]
sizes_3d = [(16,16,16), (32,32,32), (64,64,64)]

println("\n=== R2C FFT Benchmarks ===")
for dims in sizes_2d
    println("\n2D R2C FFT")
    input = rand(Float64, dims...)
    run_benchmark(input, ntuple(i -> i == 1 ? RFFT() : FFT(), 2), ntuple(identity, 2))
end

for dims in sizes_3d
    println("\n3D R2C FFT")
    input = rand(Float64, dims...)
    run_benchmark(input, ntuple(i -> i == 1 ? RFFT() : FFT(), 3), ntuple(identity, 3))
end

println("\n=== C2C FFT Benchmarks ===")
for dims in sizes_2d
    println("\n2D C2C FFT")
    input = rand(ComplexF64, dims...)
    run_benchmark(input, ntuple(i -> FFT(), 2), ntuple(identity, 2))
end

for dims in sizes_3d
    println("\n3D C2C FFT")
    input = rand(ComplexF64, dims...)
    run_benchmark(input, ntuple(i -> FFT(), 3), ntuple(identity, 3))
end

println("\n=== R2R FFT Benchmarks ===")
for dims in sizes_2d
    println("\n2D R2R FFT")
    input = rand(Float64, dims...)
    run_benchmark(input, ntuple(i -> R2R(FFTW.REDFT10), 2), ntuple(identity, 2))
end

for dims in sizes_3d
    println("\n3D R2R FFT")
    input = rand(Float64, dims...)
    run_benchmark(input, ntuple(i -> R2R(FFTW.REDFT10), 3), ntuple(identity, 3))
end