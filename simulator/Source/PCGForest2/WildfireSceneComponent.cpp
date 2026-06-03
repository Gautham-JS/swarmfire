// Fill out your copyright notice in the Description page of Project Settings.


#include "WildfireSceneComponent.h"

#include "NiagaraFunctionLibrary.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/World.h"


// Sets default values for this component's properties
UWildfireSceneComponent::UWildfireSceneComponent()
{
	// Set this component to be initialized when the game starts, and to be ticked every frame.  You can turn these features
	// off to improve performance if you don't need them.
	PrimaryComponentTick.bCanEverTick = false;

}


// Called when the game starts
void UWildfireSceneComponent::BeginPlay()
{
	Super::BeginPlay();
	SpawnRandomFires();
	
}


void UWildfireSceneComponent::SpawnRandomFires() {
    ClearAllFires();

    if (GEngine)
        GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Orange,
            FString::Printf(TEXT("SpawnRandomFires called - seeds: %d"), num_fire_seeds));

    if (!fire_niagara_system || !pcg_volume_actor)
    {
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Red,
                TEXT("[FIRE ERROR] Missing NiagaraSystem or PCGVolumeActor"));
        return;
    }


    UWorld* world = GetWorld();
    if (!world) return;

    std::vector<FVector> locations;

    for (int32 i = 0; i < num_fire_seeds; ++i) {
        // Seed location
        FVector seed_loc = GetRandomPointInVolume();

        /*if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Yellow,
                FString::Printf(TEXT("[FIRE] Seed %d location: X=%.0f Y=%.0f Z=%.0f"),
                    i, seed_loc.X, seed_loc.Y, seed_loc.Z));*/

        // Snap to ground via line trace
        FHitResult hit;
        FVector trace_end = seed_loc - FVector(0, 0, 10000.f);
        if (world->LineTraceSingleByChannel(hit, seed_loc, trace_end, ECC_WorldStatic)) {
            seed_loc = hit.ImpactPoint;
           /* if (GEngine)
                GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Cyan, FString::Printf(TEXT("[FIRE] Ground hit at Z=%.0f"), seed_loc.Z));*/
        }
        else {
            if (GEngine)
                GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Red,
                    TEXT("[FIRE] Line trace missed ground entirely"));
        }

        // Spawn 1 primary + N spread emitters per seed
        int32 spread_count = FMath::RandRange(1, 4);
        for (int32 j = 0; j <= spread_count; ++j) {
            FVector spawn_loc = seed_loc;
            if (j > 0) {
                // Random offset within spread radius
                FVector2D offset_2d = FMath::RandPointInCircle(spread_radius);
                spawn_loc += FVector(offset_2d.X, offset_2d.Y, 0.f);

                // Re-snap spread points to ground
                FHitResult spread_hit;
                FVector spread_end = spawn_loc - FVector(0, 0, 10000.f);
                if (world->LineTraceSingleByChannel(spread_hit, spawn_loc, spread_end, ECC_WorldStatic))
                    spawn_loc = spread_hit.ImpactPoint;
            }

            UNiagaraComponent* fire_comp = UNiagaraFunctionLibrary::SpawnSystemAtLocation(
                world,
                fire_niagara_system,
                spawn_loc,
                FRotator::ZeroRotator,
                FVector(1.f),
                true,   // auto-destroy = false via persistent lifetime; set true for fire-and-forget tests
                true,
                ENCPoolMethod::None
            );

            if (fire_comp) {
                // Random scale for visual variance
                float scale = FMath::FRandRange(emitter_scale_range.X, emitter_scale_range.Y);
                fire_comp->SetWorldScale3D(FVector(scale));
                active_fire_components.Add(fire_comp);
            }
            
            float scale = FMath::FRandRange(emitter_scale_range.X, emitter_scale_range.Y);
            fire_comp->SetWorldScale3D(FVector(scale * 10.0f)); 

            locations.push_back(spawn_loc);
        }
    }

    TArray<AActor*> drones;
    UGameplayStatics::GetAllActorsOfClass(GetWorld(), ADroneParent::StaticClass(), drones);
    ADroneParent* drone = nullptr;
    
    if (drones.Num() > 0) {
        drone = Cast<ADroneParent>(drones[0]);
        drone->ClearFireStencilMarkers(); // clear old markers first
    }
    else {
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Red,
                TEXT("[FIRE] No drones to call SPawn fire stencil"));
    }
    

    for (FVector loc : locations) {
        if (drone) {
            drone->SpawnFireStencilMarker(loc);
        }
    }
}

// Called every frame
void UWildfireSceneComponent::TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	// ...
}


void UWildfireSceneComponent::ClearAllFires() {
    for (UNiagaraComponent* comp : active_fire_components) {
        if (comp && comp->IsValidLowLevel())
            comp->DestroyComponent();
    }
    active_fire_components.Empty();
}


FVector UWildfireSceneComponent::GetRandomPointInVolume() const
{
    FVector origin, extent;
    pcg_volume_actor->GetActorBounds(false, origin, extent);

    return FVector(
        FMath::FRandRange(origin.X - extent.X, origin.X + extent.X),
        FMath::FRandRange(origin.Y - extent.Y, origin.Y + extent.Y),
        origin.Z + extent.Z  // top of volume so line trace downward hits terrain
    );
}

TArray<FVector> UWildfireSceneComponent::GetFireLocations() const {
    TArray<FVector> locs;
    for (const UNiagaraComponent* comp : active_fire_components) {
        if (comp && comp->IsValidLowLevel())
            locs.Add(comp->GetComponentLocation());
    }
    return locs;
}

