/*
	Class :		DroneParent
	Desc:
		Parents the BP_Drone blueprint, the main entrypoint for behavioural planning in the engine.
		Implements logic for the drone mobility and level behaviour
		Also implements operational modes for:
			1. Patrol Mode: Synthetic Data Gen by moving in a grid and sampling data, once complete, reset map and repeat.
			2. RL Train Mode: Spawns a drone in the middle of PCG & establish WS comms with DRL control server & establish control loop.

	Author:		Gautham J Suveny
	Email:		gauthamjs56@gmail.com
*/

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/Actor.h"

#include "Camera/CameraComponent.h"
#include "Components/SceneCaptureComponent2D.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialInstanceDynamic.h"

#include "EngineUtils.h"

#include "PCGComponent.h"
#include "PCGGraph.h"

#include "RLBridgeManager.h"
#include "UWildfireManager.h"

#include "DroneParent.generated.h"

class AStaticMeshActor;

UCLASS()
class PCGFOREST2_API ADroneParent : public APawn {
	GENERATED_BODY()

public:

	// Fire stencil markers
	UPROPERTY()
	TArray<AStaticMeshActor*> fire_stencil_markers;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Segmentation")
	UStaticMesh* fire_marker_mesh;

	// Frame saving
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DataCapture")
	int32 capture_every_n_steps = 10;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "DataCapture")
	UTextureRenderTarget2D* rgb_render_target;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Segmentation")
	USceneCaptureComponent2D* seg_cap;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Segmentation")
	UTextureRenderTarget2D* seg_render_target;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Segmentation")
	UMaterialInterface* segmentation_material;

	UPROPERTY(BlueprintReadOnly, Category = "Components")
	UCameraComponent* camera_down;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Camera")
	UMaterialInterface* volume_mask_base_material;

	UPROPERTY()
	UMaterialInstanceDynamic* rt_volumetric_mask_material;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="PCG")
	AActor* pcg_volume_actor;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG")
	FVector2D normalized_pcg_position;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG")
	bool is_inside_pcg = false;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	USceneCaptureComponent2D* down_cap;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category="Movement")
	float speed = 500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	float damping_fac = 1.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Movement")
	float fixed_elevation = 2800.0f;

	// Grid patrol
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Patrol")
	float grid_step_size = 1000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Patrol")
	float waypoint_reach_threshold = 100.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Patrol")
	float patrol_speed = 300.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Patrol")
	bool patrol_active = false;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "PCG")
	bool randomize_pcg_seed = true;


	// DRL properties:

	// ---- PCG Local Frame ----
	// All outputs treat PCG volume centre as (0,0)
	// and volume extents as the coordinate bounds

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG|Frame")
	FVector2D pcg_local_position;          // metres from volume centre

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG|Frame")
	FVector2D pcg_normalized_position;     // [-1, 1] from volume centre

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG|Frame")
	FVector2D pcg_unorm_position;          // [0, 1] top-left origin

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG|Frame")
	FVector2D pcg_volume_size;             // full XY size of volume in UU

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG|Frame")
	FVector2D pcg_volume_centre;           // world XY of volume centre

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG|Frame")
	FVector2D pcg_velocity_local;          // velocity in PCG frame (UU/s)

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "PCG|Frame")
	FVector2D pcg_velocity_normalized;     // velocity normalized by volume size

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "PCG|Frame")
	FVector2D WorldToPCGLocal(FVector2D world_xy) const;							// Convert any world XY position to PCG local frame

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "PCG|Frame")				// Convert any world XY position to PCG normalized [-1,1]
	FVector2D WorldToPCGNormalized(FVector2D world_xy) const;

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "PCG|Frame")				// Convert any world XY position to PCG unorm [0,1]
	FVector2D WorldToPCGUnorm(FVector2D world_xy) const;

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "PCG|Frame")				// Convert PCG normalized [-1,1] back to world XY
	FVector2D PCGNormalizedToWorld(FVector2D pcg_normalized) const;

	UFUNCTION(BlueprintCallable, BlueprintPure, Category = "PCG|Frame")
	FVector2D PCGUnormToWorld(FVector2D pcg_unorm) const;							// Convert PCG unorm [0,1] back to world XY

	UFUNCTION(BlueprintCallable, Category = "PCG|Frame")
	FString GetRLObservationJSON() const;											// Build the full RL observation dict as JSON string

	UFUNCTION(BlueprintCallable, Category = "DataCapture")
	void SaveCurrentFrame();

	UFUNCTION(BlueprintCallable, Category = "Patrol")
	void StartGridPatrol();

	UFUNCTION(BlueprintCallable, Category = "Patrol")
	void StopPatrol();

	ADroneParent();																	// Self explanatory constructor, duh

	void SpawnFireStencilMarker(FVector location);									// Spawn marker meshes for fire as stenciling fails on particle systems
	void ClearFireStencilMarkers();													// Clean up markers
	void RandomizePCGSeed();														// Randomize seed for PCG system

	void Step();																	// Steps per timestep, should be called from env_handler's tick or when step is called there via WS
	void Reset();																	// Resets drone to start pos, should be called from env_handler's Reset impl.


protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

private:

	FString session_id;

	bool is_x_moving = false;
	bool is_y_moving = false;
	float x_velocity = 0.0f;
	float y_velocity = 0.0f;

	int32 tick_counter = 0;
	int32 frame_save_index = 0;

	TArray<FVector> patrol_waypoints;
	int32 current_waypoint_index = 0;
	bool patrol_complete = false;

	// Cached volume bounds — updated in UpdateVolumeAwareness
	FVector2D pcg_vol_min;
	FVector2D pcg_vol_max;
	FVector2D pcg_vol_extent;
	bool pcg_frame_valid = false;

	void UpdatePCGFrame();

	void GenerateGridWaypoints();											// Uniform samples waypoints for patroling mode
	void TickPatrol(float DeltaTime);										// Child func of tick for patroling mode


	void UpdateVolumeAwareness();											// Coordinate transform from world -> PCG Volume.
	void AutoAssignPCGVolume();												// Assign PCG actor to class level var

	UFUNCTION()
	void OnPCGGenerationComplete(UPCGComponent* comp);

public:	
	virtual void Tick(float DeltaTime) override;							// Called every frame
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;					// Called to bind functionality to input

	void SetTreeMeshStencilIDs();											// Assign stencil IDs to segment tree meshes
	void SetPCGVolume(AActor* pcg_actor);


	void MoveAhead();
	void MoveBack();
	void MoveLeft();
	void MoveRight();
	void StopX();
	void StopY();

};
