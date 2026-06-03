// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "Components/SceneComponent.h"

#include "NiagaraSystem.h"
#include "NiagaraComponent.h"
#include "Kismet/GameplayStatics.h"
#include "DroneParent.h"

#include "WildfireSceneComponent.generated.h"


UCLASS( ClassGroup=(Custom), meta=(BlueprintSpawnableComponent) )
class PCGFOREST2_API UWildfireSceneComponent : public USceneComponent
{
	GENERATED_BODY()

public:	
	// Sets default values for this component's properties
	UWildfireSceneComponent();

    // Assign your Niagara fire asset in Blueprint/Editor
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    UNiagaraSystem* fire_niagara_system;

    // How many fire sources to seed
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    int32 num_fire_seeds = 10;

    // Radius around each seed to spawn additional emitters (tree-to-tree spread sim)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    float spread_radius = 300.0f;

    // Scale range for fire emitters (visual variety)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    FVector2D emitter_scale_range = FVector2D(0.1f, 0.5f);

    // The PCG volume actor (same one your drone references)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    AActor* pcg_volume_actor;

    // Call this to randomize and respawn fire (e.g. per training episode)
    UFUNCTION(BlueprintCallable, Category = "Fire")
    void SpawnRandomFires();

    UFUNCTION(BlueprintCallable, Category = "Fire")
    void ClearAllFires();

    // Returns world positions of all active fire emitters (for label export)
    UFUNCTION(BlueprintCallable, Category = "Fire")
    TArray<FVector> GetFireLocations() const;

protected:
	// Called when the game starts
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void TickComponent(float DeltaTime, ELevelTick TickType, FActorComponentTickFunction* ThisTickFunction) override;

private:
    UPROPERTY()
    TArray<UNiagaraComponent*> active_fire_components;

    FVector GetRandomPointInVolume() const;

		
};
