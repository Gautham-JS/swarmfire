// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "NiagaraSystem.h"
#include "NiagaraComponent.h"
#include "UWildfireManager.generated.h"

UCLASS()
class PCGFOREST2_API AUWildfireManager : public AActor
{
	GENERATED_BODY()
	
public:	
	// Sets default values for this actor's properties
	AUWildfireManager();

    // Assign your Niagara fire asset in Blueprint/Editor
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    UNiagaraSystem* fire_niagara_system;

    // How many fire sources to seed
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    int32 num_fire_seeds = 5;

    // Radius around each seed to spawn additional emitters (tree-to-tree spread sim)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    float spread_radius = 400.0f;

    // Scale range for fire emitters (visual variety)
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Fire")
    FVector2D emitter_scale_range = FVector2D(0.5f, 2.0f);

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
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

private:
    UPROPERTY()
    TArray<UNiagaraComponent*> active_fire_components;

    FVector GetRandomPointInVolume() const;

};
