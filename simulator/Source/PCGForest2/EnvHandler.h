// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"

#include "WebsocketManager.h"
#include "DroneParent.h"
#include "PCGComponent.h"


#include "EnvHandler.generated.h"

UCLASS()
class PCGFOREST2_API AEnvHandler : public AActor {
	GENERATED_BODY()
	
public:	

	UPROPERTY(EditAnywhere)
	TSubclassOf<ADroneParent> drone_class;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	bool auto_connect_ws = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "RL")
	FString rl_server_url = TEXT("ws://localhost:8080/");

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	FVector drone_spawn_location = FVector(0, 0, 2800);

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Environment")
	FRotator drone_spawn_rotation = FRotator::ZeroRotator;

	// Runtime stuff
	ADroneParent* drone_instance = nullptr;												// ptr to drone instance object or drone BP class

	UPROPERTY(BlueprintReadOnly, Category = "Environment")
	bool environment_ready = false;

	UPROPERTY(BlueprintReadOnly, Category = "Environment")
	bool waiting_for_action = false;

	// Sets default values for this actor's properties
	AEnvHandler();

	void ResetEnv();								// Reset: RL func 
	void CloseEnv();								// Close: RL func
	void StepEnv(const FString& action_json);		// Step: RL func

	void DestroyDrone();							// Clean up drone instance
	void SpawnDrone();								// Spawn the level ideally, ig the drone class also works??
	void RegeneratePCG();							// Regenerate PCG volumes & generator seeding


	UFUNCTION(BlueprintCallable, Category = "RL")
	void SendObservation(bool done = false);		// Set up observation & flush to web-socket interface
	
private:
	UPROPERTY()
	UPCGComponent* pcg_component = nullptr;

	void FindPCGComponent();							// Finds the PCG volume in the space
	void SetupWebsocket();								// Set up WS connections
	void HandleWebsocketMessage(const FString& msg);	// Handle message coming from WS. Should be async, runs in separate thread
	void OnPCGGenerated(UPCGComponent* comp);			// PCG ready event, for stencil assignment

	TSharedPtr<FJsonObject> ParseJSON(const FString& json);

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;
	
public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	virtual void EndPlay(const EEndPlayReason::Type reason) override;

	void SpawnFireStencilMarker(FVector location);									// Spawn marker meshes for fire as stenciling fails on particle systems
	void ClearFireStencilMarkers();													// Clean up markers
	void RandomizePCGSeed();														// Randomize seed for PCG system
	void SetTreeMeshStencilIDs();													// Set Stencil IDs for tree meshes.

};
