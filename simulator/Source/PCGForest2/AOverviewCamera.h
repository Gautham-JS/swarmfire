// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Components/StaticMeshComponent.h"
#include "WildfireSceneComponent.h"

#include "AOverviewCamera.generated.h"

UCLASS()
class PCGFOREST2_API AAOverviewCamera : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AAOverviewCamera();

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Fire")
    UWildfireSceneComponent* wildfire_component;

    // Assign your PCG volume here (same ref as drone)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Overview")
    AActor* pcg_volume_actor;

    // How high above the volume centre to sit
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Overview")
    float camera_height_offset = 8000.0f;

    // The render target this camera writes to (create in editor, assign here)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Overview")
    UTextureRenderTarget2D* overview_render_target;

    // The drone actor to track and mark
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Overview")
    AActor* drone_actor;

    // Marker mesh (assign a simple sphere/cone in editor)
    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Overview")
    UStaticMeshComponent* drone_marker;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Overview")
    USceneCaptureComponent2D* overview_cap;

    // Scale of the drone marker (tweak per scene scale)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Overview")
    float marker_scale = 80.0f;

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
private:
    void SnapToPCGVolumeCentre();
    void AutoAssignPCGVolume();

};
