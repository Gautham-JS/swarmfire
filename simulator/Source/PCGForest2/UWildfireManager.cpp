// Fill out your copyright notice in the Description page of Project Settings.


#include "UWildfireManager.h"

#include "NiagaraFunctionLibrary.h"
#include "Kismet/KismetMathLibrary.h"
#include "Engine/World.h"

// Sets default values
AUWildfireManager::AUWildfireManager()
{
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = false;

}

// Called when the game starts or when spawned
void AUWildfireManager::BeginPlay()
{
	Super::BeginPlay();
	SpawnRandomFires();
}

// Called every frame
void AUWildfireManager::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

}

void AUWildfireManager::SpawnRandomFires() {
    ClearAllFires();

    if (!fire_niagara_system || !pcg_volume_actor) {
        UE_LOG(LogTemp, Warning, TEXT("WildfireManager: Missing NiagaraSystem or PCGVolumeActor"));
        return;
    }

    UWorld* world = GetWorld();
    if (!world) return;

    for (int32 i = 0; i < num_fire_seeds; ++i) {
        // Seed location
        FVector seed_loc = GetRandomPointInVolume();

        // Snap to ground via line trace
        FHitResult hit;
        FVector trace_end = seed_loc - FVector(0, 0, 10000.f);
        if (world->LineTraceSingleByChannel(hit, seed_loc, trace_end, ECC_WorldStatic))
            seed_loc = hit.ImpactPoint;

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
        }
    }
}


void AUWildfireManager::ClearAllFires() {
    for (UNiagaraComponent* comp : active_fire_components) {
        if (comp && comp->IsValidLowLevel())
            comp->DestroyComponent();
    }
    active_fire_components.Empty();
}


FVector AUWildfireManager::GetRandomPointInVolume() const
{
    FVector origin, extent;
    pcg_volume_actor->GetActorBounds(false, origin, extent);

    return FVector(
        FMath::FRandRange(origin.X - extent.X, origin.X + extent.X),
        FMath::FRandRange(origin.Y - extent.Y, origin.Y + extent.Y),
        origin.Z + extent.Z  // top of volume so line trace downward hits terrain
    );
}

TArray<FVector> AUWildfireManager::GetFireLocations() const {
    TArray<FVector> locs;
    for (const UNiagaraComponent* comp : active_fire_components)
    {
        if (comp && comp->IsValidLowLevel())
            locs.Add(comp->GetComponentLocation());
    }
    return locs;
}
