using TensorTrains
using Test
using Random

function ttcheckerr(Y1, Y2)
    return norm(add(1.0, Y1, -1.0, Y2)) < length(Y1) * eps(Float64)*max(norm(Y1),norm(Y2))
end

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
    @testset "matvec" begin
        Random.seed!(1234);
        d = 8;
        Id = tteye(d);
        T1 = ttrandn(d);
        @test norm(T1) ≈ norm(matvec(Id,T1))
        Y1 = matvec(add(1.0,Id,1.0,Id),T1);
        Y2 = matvec(scale(Id,2.0),T1);
        @test ttcheckerr(Y1,Y2);
        A1 = deepcopy(Id);
        add!(A1, 2.5, Id);
        Y1 = matvec(A1, T1);
        A2 = deepcopy(Id);
        scale!(A2, 3.5);
        Y2 = matvec(A2, T1);
        @test ttcheckerr(Y1,Y2);
    end
    @testset "matmat" begin
        Random.seed!(1234);
        d = 8;
        Id = tteye(d);
        A1 = deepcopy(Id);
        scale!(A1,Float64(π));
        T1 = ttrandn(d);
        A2 = matmat(Id,A1);
        Y1 = matvec(A1,T1);
        Y2 = matvec(A2,T1);
        @test ttcheckerr(Y1,Y2);
        A2 = matmat(A2,Id);
        Y2 = matvec(A2,T1);
        @test ttcheckerr(Y1,Y2);
    end
    @testset "hadamard" begin
        Random.seed!(1234);
        T1 = ttones(6);
        T2 = ttrandn(6);
        @test norm(hadamard(T1,T2)) ≈ norm(T2)
    end
    @testset "kron" begin
        Random.seed!(1234);
        A1 = tteye(8);
        A2 = tteye(8);
        T1 = ttrandn(8);
        T2 = ttrandn(8);
        scale!(A1, 0.5);
        scale!(A2, Float64(π));
        Y1 = matvec(ttkron(A1,A2),ttkron(T1,T2));
        Y2 = ttkron(matvec(A1,T1),matvec(A2,T2));
        @test ttcheckerr(Y1,Y2);
    end
end;
