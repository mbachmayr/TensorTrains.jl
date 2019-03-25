using TensorTrains
using Test
using Random

@testset "TensorTrains" begin
    @testset "dot prod" begin
        T1 = ttones(3); T2 = ttones(3)
        @test dot(T1,T2) ≈ 8.0
        T1 = ttrandn(5); T2 = ttrandn(5)
        @test dot(T1,T2) ≈ vec(decompress(T1))'*vec(decompress(T2))
        @test dot(T1,T2) ≈ vec(unfold(T1))'*vec(unfold(T2))
    # @test dot(ttones)
    end;
    @testset "norm" begin
        T1 = ttones(3);
        @test norm(T1) ≈ √8
        T1 = ttrandn(5);
        @test norm(T1)  ≈ √dot(T1,T1)
        @test norm!(T1) ≈ √dot(T1,T1)
    end;
    @testset "add" begin
        T1 = ttones(3);
        add!(T1, ttones(3));
        @test norm(T1) ≈ 2*√8;
        add!(T1, -1.0, ttones(3));
        @test norm(T1) ≈ √8;
        T2 = add(2.0, T1, -1.0, T1);
        @test norm(T2) ≈ norm(T1);
    end
    @testset "orthogonality" begin
        T1 = ttones(4);
        add!(T1, ttones(4));
        add!(T1, ttones(4));
        rightOrth!(T1);
        @test norm(T1[end]) ≈ norm(T1);
        leftOrth!(T1);
        @test norm(T1[1]) ≈ norm(T1);
    end
    @testset "svd" begin
        T1 = ttones(4);
        add!(T1, ttones(4));
        add!(T1, ttones(4));
        svdtrunc!(T1, svd!(T1), 1e-16);
        @test ranks(T1) == [1,1,1];
        Random.seed!(1234);
        T1 = ttrandn(8);
        add!(T1, ttrandn(8));
        add!(T1, ttrandn(8));
        T2 = deepcopy(T1);
        svdtrunc!(T1, svd!(T1), 1e-1);
        @test norm(add(1.0, T1, -1.0, T2)) < 2e-1;
        T1 = deepcopy(T2);
        svdtrunc!(T1, svd!(T1), 2e-1);
        @test norm(add(1.0, T1, -1.0, T2)) < 4e-1;
    end
end;
