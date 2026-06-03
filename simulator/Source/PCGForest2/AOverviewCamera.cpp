// Fill out your copyright notice in the Description page of Project Settings.


#include "AOverviewCamera.h"

#include "Kismet/GameplayStatics.h"

// Sets default values
AAOverviewCamera::AAOverviewCamera()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

    USceneComponent* root = CreateDefaultSubobject<USceneComponent>(TEXT("Root"));
    RootComponent = root;

    this->wildfire_component = CreateDefaultSubobject<UWildfireSceneComponent>(TEXT("WildfireComponent"));
    this->wildfire_component->SetupAttachment(RootComponent);

    this->overview_cap = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("OverviewCapture"));
    this->overview_cap->SetupAttachment(RootComponent);
    this->overview_cap->SetRelativeRotation(FRotator(-90.0f, 0.0f, 0.0f)); // straight down
    this->overview_cap->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
    this->overview_cap->bCaptureEveryFrame = true;
    this->overview_cap->SetActive(true);

    // Orthographic projection fits a top-down overview much better than perspective
    this->overview_cap->ProjectionType = ECameraProjectionMode::Orthographic;
    this->overview_cap->OrthoWidth = 30000.0f; // adjust to match your volume size

    // Drone marker — assign a mesh in editor (e.g. engine Sphere or Cone)
    this->drone_marker = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("DroneMarker"));
    this->drone_marker->SetupAttachment(RootComponent);
    this->drone_marker->SetCollisionEnabled(ECollisionEnabled::NoCollision);
    this->drone_marker->SetCastShadow(false);

}

// Called when the game starts or when spawned
void AAOverviewCamera::BeginPlay()
{
	Super::BeginPlay();

    if (this->overview_render_target)
        this->overview_cap->TextureTarget = this->overview_render_target;
    else {
        GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Red,
            FString::Printf(TEXT("[ERROR] : Overview Actor overview render target is NULL")));
    }

    this->SnapToPCGVolumeCentre();
    this->AutoAssignPCGVolume();
}


void AAOverviewCamera::AutoAssignPCGVolume() {
    // Auto-find PCG volume by tag if not manually assigned
    if (!this->pcg_volume_actor)
    {
        TArray<AActor*> tagged_actors;
        UGameplayStatics::GetAllActorsWithTag(
            GetWorld(),
            FName("PCGVolume"),
            tagged_actors
        );

        if (tagged_actors.Num() > 0)
        {
            this->pcg_volume_actor = tagged_actors[0];
            if (GEngine)
                GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green,
                    TEXT("[Drone] PCG Volume auto-assigned"));
        }
        else
        {
            if (GEngine)
                GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
                    TEXT("[Drone] ERROR: No actor with tag PCGVolume found"));
        }
    }

}

// Called every frame
void AAOverviewCamera::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

    if (!drone_actor) return;

    FVector drone_world_loc = drone_actor->GetActorLocation();

    // Move the marker to match the drone's XY, but pin Z just below this camera
    // so it's always visible in the capture without occluding the scene
    FVector marker_world = drone_world_loc;
    marker_world.Z = GetActorLocation().Z - 100.0f;
    this->drone_marker->SetWorldLocation(marker_world);
    this->drone_marker->SetWorldScale3D(FVector(marker_scale / 100.0f));

}

void AAOverviewCamera::SnapToPCGVolumeCentre()
{
    if (!pcg_volume_actor) {
        GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
            FString::Printf(TEXT("[ERROR] : Overview Actor PCG volume actor is null")));
        return;
    }

    FVector origin, extent;
    pcg_volume_actor->GetActorBounds(false, origin, extent);

    // Sit directly above the volume centre
    FVector cam_loc = origin;
    cam_loc.Z = origin.Z + extent.Z + camera_height_offset;
    SetActorLocation(cam_loc);

    // Auto-fit ortho width to the volume's largest horizontal dimension
    float fit_width = FMath::Max(extent.X, extent.Y) * 2.0f * 1.1f; // 10% padding
    this->overview_cap->OrthoWidth = fit_width;

    this->overview_cap->bCaptureEveryFrame = false;
    this->overview_cap->CaptureScene(); // manual trigger
    this->overview_cap->bCaptureEveryFrame = true;

    if (GEngine)
    {
        GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Cyan,
            FString::Printf(TEXT("Overview cam placed at Z=%.0f, OrthoWidth=%.0f"),
                cam_loc.Z, fit_width));
    }
}

