// Fill out your copyright notice in the Description page of Project Settings.


#include "DroneParent.h"
#include "WebsocketManager.h"

#include "Kismet/GameplayStatics.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/HierarchicalInstancedStaticMeshComponent.h"
#include "UObject/UObjectIterator.h"
#include "EngineUtils.h"

#include "Engine/TextureRenderTarget2D.h"
#include "PCGComponent.h"

#include "ImageUtils.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"

#include "Json.h"
#include "JsonUtilities.h"
#include "Dom/JsonValue.h"

#include "Dom/JsonObject.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"



// Sets default values
ADroneParent::ADroneParent() {
	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	AutoPossessPlayer = EAutoReceiveInput::Player0;

	UE_LOG(LogTemp, Warning, TEXT("[DRONE] Constructing"));


	USceneComponent* SceneRoot = CreateDefaultSubobject<USceneComponent>(TEXT("SceneRoot"));
	RootComponent = SceneRoot;

	this->down_cap = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("DownwardCapture"));
	this->down_cap->SetupAttachment(RootComponent);
	this->down_cap->SetWorldRotation(FRotator(-90.0f, 0.0f, 0.0f));
	this->down_cap->CaptureSource = ESceneCaptureSource::SCS_FinalColorHDR;
	this->down_cap->bCaptureEveryFrame = true;
	this->down_cap->SetActive(true);

	this->seg_cap = CreateDefaultSubobject<USceneCaptureComponent2D>(TEXT("SegCapture"));
	this->seg_cap->SetupAttachment(RootComponent);
	this->seg_cap->SetRelativeRotation(FRotator(-90.0f, 0.0f, 0.0f));
	this->seg_cap->CaptureSource = ESceneCaptureSource::SCS_FinalColorLDR;
	this->seg_cap->bCaptureEveryFrame = true;
	this->seg_cap->SetActive(true);

}

void ADroneParent::RandomizePCGSeed() {
	if (!this->pcg_volume_actor) return;

	UPCGComponent* pcg_comp = this->pcg_volume_actor->FindComponentByClass<UPCGComponent>();
	if (!pcg_comp) return;

	int32 new_seed = FMath::RandRange(0, TNumericLimits<int32>::Max());
	pcg_comp->Seed = new_seed;

	// Bind to generation complete delegate
	pcg_comp->OnPCGGraphGeneratedDelegate.AddUObject(this, &ADroneParent::OnPCGGenerationComplete);

	pcg_comp->DirtyGenerated();
	pcg_comp->Refresh();
}

void ADroneParent::OnPCGGenerationComplete(UPCGComponent* comp) {
	// Unbind so it doesn't fire multiple times
	comp->OnPCGGraphGeneratedDelegate.RemoveAll(this);

	if (GEngine)
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, TEXT("[PCG] Generation complete, assigning stencils"));

	this->SetTreeMeshStencilIDs();
	this->StartGridPatrol();
}


// Called when the game starts or when spawned
void ADroneParent::BeginPlay() {
    Super::BeginPlay();
    UE_LOG(LogTemp, Warning, TEXT("[DRONE] <BeginPlay> - Enter"));
	if (this->rgb_render_target) {
		this->down_cap->TextureTarget = this->rgb_render_target;
		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, FString::Printf(TEXT("down_cap RT assigned: %s"), *this->rgb_render_target->GetName()));
	}
	else {
		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("[ERROR] rgb_render_target is NULL"));
	}

	FVector start_loc = GetActorLocation();
	start_loc.Z = this->fixed_elevation;
	SetActorLocation(start_loc);

	TArray<UCameraComponent*> cameras;
	GetComponents<UCameraComponent>(cameras);

	//this->down_cap->SetActive(true);

	if (cameras.Num() > 0) {
		APlayerController* pc = Cast<APlayerController>(GetController());
		if (pc) {
			// Deactivate all first
			for (UCameraComponent* c : cameras) {
				c->SetActive(false);
			}

			// Find CameraDown by name — print all names to debug
			for (UCameraComponent* c : cameras) {
				if (GEngine) {
					GEngine->AddOnScreenDebugMessage(-1, 30.0f, FColor::Red, FString::Printf(TEXT("Camera found: %s"), *c->GetName()));
				}
				if (c->GetName().Contains("Downward")) {
					c->SetActive(true);
					pc->SetViewTargetWithBlend(this, 0.0f);
				}
			}
		}
	}

	if (this->seg_render_target)
		this->seg_cap->TextureTarget = this->seg_render_target;

	// Create output directory
	FString save_dir = FPaths::ProjectSavedDir() + TEXT("WildfireCaptures/");
	IFileManager::Get().MakeDirectory(*save_dir, true);

	if (GEngine)
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, FString::Printf(TEXT("Saving frames to: %s"), *save_dir));

	UE_LOG(LogTemp, Warning, TEXT("[DRONE] RT assignment complete"));
	//WebsocketManager::Clean();
	//WebsocketManager::Get()->Connect("ws://localhost:8080/");
	//WebsocketManager::Get()->SendJsonMessage("session_start", "{msg: 'starting session'}");
	

	FDateTime now = FDateTime::Now();
	this->session_id = FString::Printf(
		TEXT("%04d%02d%02d_%02d%02d%02d"),
		now.GetYear(), now.GetMonth(), now.GetDay(),
		now.GetHour(), now.GetMinute(), now.GetSecond()
	);

	if (GEngine)
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, FString::Printf(TEXT("Session ID: %s"), *this->session_id));

	UE_LOG(LogTemp, Warning, TEXT("[DRONE] Calling autoassign PCG"));
	this->AutoAssignPCGVolume();
	// Then randomize PCG seed — stencils and patrol
	// start automatically via OnPCGGenerationComplete delegate
	if (this->randomize_pcg_seed)
		this->RandomizePCGSeed();
	else {
		// If not randomizing, assign stencils and start patrol directly
		this->SetTreeMeshStencilIDs();
		this->StartGridPatrol();
	}
}


void ADroneParent::SpawnFireStencilMarker(FVector location) {
	UE_LOG(LogTemp, Warning, TEXT("[DRONE] [SpawnFireStencilMarker] enter"));
	UWorld* world = GetWorld();
	if (!world || !this->fire_marker_mesh) return;

	FActorSpawnParameters params;
	params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

	AStaticMeshActor* marker = world->SpawnActor<AStaticMeshActor>(
		AStaticMeshActor::StaticClass(),
		location,
		FRotator::ZeroRotator,
		params
	);

	if (marker) {
		UStaticMeshComponent* mc = marker->GetStaticMeshComponent();
		float sphere_default_radius = 50.0f;
		float desired_radius = 150.0f; // fire visual size
		float scale = desired_radius / sphere_default_radius;
		mc->SetWorldScale3D(FVector(scale));
		mc->SetStaticMesh(this->fire_marker_mesh);
		mc->SetWorldScale3D(FVector(3.0f));

		// Enable custom depth stencil = 3
		mc->SetRenderCustomDepth(true);
		mc->SetCustomDepthStencilValue(3);
		mc->bRenderInMainPass = false;        // invisible in main scene
		mc->bRenderInDepthPass = false;       // invisible in depth
		mc->bRenderCustomDepth = true;        // VISIBLE in custom depth only

		// Hide from RGB captures
		if (this->down_cap)
			this->down_cap->HiddenComponents.AddUnique(mc);

		fire_stencil_markers.Add(marker);
	}
	UE_LOG(LogTemp, Warning, TEXT("[DRONE] [SpawnFireStencilMarker] exit"));
}



void ADroneParent::StartGridPatrol() {
	if (!this->pcg_volume_actor) {
		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red, TEXT("[Patrol] No PCG volume assigned"));
		return;
	}

	this->patrol_waypoints.Empty();
	this->current_waypoint_index = 0;
	this->patrol_complete = false;
	this->patrol_active = true;

	GenerateGridWaypoints();

	if (this->patrol_waypoints.Num() > 0) {
		FVector first = this->patrol_waypoints[0];								// Teleport to first waypoint immediately
		first.Z = this->fixed_elevation;
		SetActorLocation(first);

		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, FString::Printf(TEXT("[Patrol] Grid started: %d waypoints"), this->patrol_waypoints.Num()));
	}
}


void ADroneParent::StopPatrol() {
	this->patrol_active = false;
	this->x_velocity = 0.0f;
	this->y_velocity = 0.0f;
	if (GEngine)
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Yellow, TEXT("[Patrol] Stopped"));
}


void ADroneParent::GenerateGridWaypoints() {
	FVector origin, extent;
	this->pcg_volume_actor->GetActorBounds(false, origin, extent);

	FVector vol_min = origin - extent;
	FVector vol_max = origin + extent;

	// Generate boustrophedon (snake/lawnmower) path
	// alternates direction each row for continuous coverage
	bool left_to_right = true;

	float y = vol_min.Y;
	while (y <= vol_max.Y) {
		if (left_to_right) {
			float x = vol_min.X;
			while (x <= vol_max.X) {
				this->patrol_waypoints.Add(FVector(x, y, this->fixed_elevation));
				x += this->grid_step_size;
			}
			this->patrol_waypoints.Add(FVector(vol_max.X, y, this->fixed_elevation));						// Add corner point at end of row to ensure full coverage
		}
		else {
			float x = vol_max.X;
			while (x >= vol_min.X) {
				this->patrol_waypoints.Add(FVector(x, y, this->fixed_elevation));
				x -= this->grid_step_size;
			}
			// Add corner point
			this->patrol_waypoints.Add(FVector(vol_min.X, y, this->fixed_elevation));
		}

		left_to_right = !left_to_right;
		y += this->grid_step_size;
	}

	if (GEngine)
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Cyan, FString::Printf(TEXT("[Patrol] Generated %d waypoints"), this->patrol_waypoints.Num()));
}

void ADroneParent::TickPatrol(float DeltaTime) {
	if (!this->patrol_active) return;
	if (this->patrol_waypoints.Num() == 0) return;


	
	// Check if patrol complete
	if (this->current_waypoint_index >= this->patrol_waypoints.Num()) {
		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green, FString::Printf(TEXT("[Patrol] Complete — %d frames saved. Restarting level..."), this->frame_save_index));

		this->patrol_active = false;

		// Short delay then restart — use a timer so the last frame saves first
		FTimerHandle restart_timer;
		GetWorldTimerManager().SetTimer(
			restart_timer,
			[this]() {
				UGameplayStatics::OpenLevel(
					this,
					FName(*UGameplayStatics::GetCurrentLevelName(this))
				);
			},
			1.0f,   // 1 second delay
			false   // don't loop
		);
		return;
	}

	FVector target = this->patrol_waypoints[this->current_waypoint_index];
	target.Z = this->fixed_elevation;

	FVector current_loc = GetActorLocation();
	FVector to_target = target - current_loc;
	to_target.Z = 0.0f;

	float dist = to_target.Size();

	if (dist < this->waypoint_reach_threshold) {
		// Reached waypoint — snap to it and advance
		FVector snapped = target;
		snapped.Z = this->fixed_elevation;
		SetActorLocation(snapped);

		this->x_velocity = 0.0f;
		this->y_velocity = 0.0f;
		this->current_waypoint_index++;

		if (GEngine)
			GEngine->AddOnScreenDebugMessage(20, 2.f, FColor::Cyan, FString::Printf(TEXT("[Patrol] Waypoint %d / %d"), this->current_waypoint_index, this->patrol_waypoints.Num()));
	}
	else
	{
		// Move toward waypoint
		FVector direction = to_target.GetSafeNormal();
		this->x_velocity = direction.X * this->patrol_speed;
		this->y_velocity = direction.Y * this->patrol_speed;
	}
}



void ADroneParent::ClearFireStencilMarkers() {
	for (AStaticMeshActor* marker : fire_stencil_markers) {
		if (marker) marker->Destroy();
	}
	fire_stencil_markers.Empty();
}

void ADroneParent::SaveCurrentFrame() {

	FString save_dir = FString::Printf(
		TEXT("E:/Datasets/Thesis/UE5Wildfire/%s/"),
		*this->session_id
	);
	IFileManager::Get().MakeDirectory(*save_dir, true);

	// Helper to read pixels from a render target
	auto ReadRT = [](UTextureRenderTarget2D* rt, TArray<FColor>& out_pixels) -> bool {
		if (!rt) return false;
		FTextureRenderTargetResource* res = rt->GameThread_GetRenderTargetResource();
		if (!res) return false;
		return res->ReadPixels(out_pixels);
	};

	// Helper to save a TArray<FColor> as PNG
	auto SaveAsPNG = [](TArray<FColor>& pixels, int32 w, int32 h, FString path) -> bool {
		for (FColor& p : pixels) p.A = 255;
		TArray<uint8> png_data;
		FImageUtils::CompressImageArray(w, h, pixels, png_data);
		return FFileHelper::SaveArrayToFile(png_data, *path);
	};

	// --- Save RGB frame ---
	TArray<FColor> rgb_pixels;
	if (ReadRT(this->rgb_render_target, rgb_pixels)) {
		FString path = FString::Printf(TEXT("%srgb_%05d.png"),
			*save_dir, this->frame_save_index);

		if (SaveAsPNG(rgb_pixels,
			this->rgb_render_target->SizeX,
			this->rgb_render_target->SizeY, path)) {
			if (GEngine)
				GEngine->AddOnScreenDebugMessage(-1, 2.f, FColor::Green, FString::Printf(TEXT("Saved RGB: rgb_%05d.png"), this->frame_save_index));
		}
	}
	else {
		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 2.f, FColor::Red, TEXT("[ERROR] Failed to read RGB render target"));
	}

	// --- Read segmentation render target once ---
	TArray<FColor> seg_pixels;
	if (ReadRT(this->seg_render_target, seg_pixels)) {
		int32 w = this->seg_render_target->SizeX;
		int32 h = this->seg_render_target->SizeY;

		auto SaveBinaryMask = [&](
			FString class_name,
			TFunction<bool(const FColor&)> classifier) -> void {
				TArray<FColor> binary_pixels;
				binary_pixels.SetNum(seg_pixels.Num());

				for (int32 i = 0; i < seg_pixels.Num(); i++) {
					// White pixel if this class, black otherwise
					binary_pixels[i] = classifier(seg_pixels[i])
						? FColor(255, 255, 255, 255)
						: FColor(0, 0, 0, 255);
				}

				FString path = FString::Printf(TEXT("%smask_%s_%05d.png"), *save_dir, *class_name, this->frame_save_index);

				TArray<uint8> png_data;
				FImageUtils::CompressImageArray(w, h, binary_pixels, png_data);

				if (FFileHelper::SaveArrayToFile(png_data, *path)) {
					if (GEngine)
						GEngine->AddOnScreenDebugMessage(-1, 2.f, FColor::Cyan, FString::Printf(TEXT("Saved mask: mask_%s_%05d.png"), *class_name, this->frame_save_index));
				}
			};

		// Stencil 1 → Red dominant, low green, low blue
		SaveBinaryMask(
			TEXT("tree_a"), 
			[](const FColor& c) -> bool {return c.R > 100 && c.G < 50 && c.B < 50; }
		);

		// Stencil 2 → Green dominant
		SaveBinaryMask(
			TEXT("trees_other"), 
			[](const FColor& c) -> bool { return c.G > 100 && c.R < 50 && c.B < 50; }
		);

		// Stencil 3 → Blue dominant
		SaveBinaryMask(
			TEXT("fire"),
			[](const FColor& c) -> bool { return c.B > 100 && c.R < 50 && c.G < 50; }
		);
	}
	else {
		if (GEngine)
			GEngine->AddOnScreenDebugMessage(-1, 2.f, FColor::Red, TEXT("[ERROR] Failed to read seg render target"));
	}

	// --- Save metadata JSON ---
	FString json = FString::Printf(
		TEXT("{\n")
		TEXT("  \"frame\": %d,\n")
		TEXT("  \"drone_x\": %.2f,\n")
		TEXT("  \"drone_y\": %.2f,\n")
		TEXT("  \"drone_z\": %.2f,\n")
		TEXT("  \"norm_x\": %.4f,\n")
		TEXT("  \"norm_y\": %.4f,\n")
		TEXT("  \"inside_pcg\": %s,\n")
		TEXT("  \"stencil_map\": {\n")
		TEXT("    \"0\": \"background\",\n")
		TEXT("    \"1\": \"tree_a\",\n")
		TEXT("    \"2\": \"other_trees\",\n")
		TEXT("    \"3\": \"fire\"\n")
		TEXT("  },\n")
		TEXT("  \"fire_markers\": %d\n")
		TEXT("}\n"),
		this->frame_save_index,
		GetActorLocation().X,
		GetActorLocation().Y,
		GetActorLocation().Z,
		this->normalized_pcg_position.X,
		this->normalized_pcg_position.Y,
		this->is_inside_pcg ? TEXT("true") : TEXT("false"),
		this->fire_stencil_markers.Num()
	);

	FString json_path = FString::Printf(TEXT("%smeta_%05d.json"), *save_dir, this->frame_save_index);
	FFileHelper::SaveStringToFile(json, *json_path);

	this->frame_save_index++;
}


void ADroneParent::SetTreeMeshStencilIDs() {	
	UE_LOG(LogTemp, Warning, TEXT("[DRONE] SetTreeMeshStencilIDs -- enter"));

	UWorld* world = GetWorld();
	if (!world) return;

	int32 stencil_1_count = 0;
	int32 stencil_2_count = 0;

	for (TActorIterator<AActor> it(world); it; ++it) {
		AActor* actor = *it;
		TArray<UActorComponent*> all_comps;
		actor->GetComponents(all_comps);

		for (UActorComponent* comp : all_comps) {
			UInstancedStaticMeshComponent* ismc = Cast<UInstancedStaticMeshComponent>(comp);
			if (!ismc || !ismc->GetStaticMesh()) continue;

			FString mesh_name = ismc->GetStaticMesh()->GetName();
			ismc->SetRenderCustomDepth(true);

			if (mesh_name.Contains("Forest_04")) {
				ismc->SetCustomDepthStencilValue(1); // Red
				stencil_1_count++;
			}
			else if (mesh_name.Contains("Floor") || mesh_name.Contains("Landscape"))
				ismc->SetCustomDepthStencilValue(0); // ignore floor
			else {
				ismc->SetCustomDepthStencilValue(2); // Green = remaining trees
				stencil_2_count++;
			}
		}
	}

	if (GEngine)
		GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Green, FString::Printf(TEXT("Stencil 1: %d components | Stencil 2: %d components"), stencil_1_count, stencil_2_count));
	UE_LOG(LogTemp, Warning, TEXT("[DRONE] SetTreeMeshStencilIDs -- exit | S1 : %d | S2 : %d"), stencil_1_count, stencil_2_count);
}


void ADroneParent::AutoAssignPCGVolume() {
	// Auto-find PCG volume by tag if not manually assigned
	if (!this->pcg_volume_actor) {
		TArray<AActor*> tagged_actors;
		UGameplayStatics::GetAllActorsWithTag(
			GetWorld(),
			FName("PCGVolume"),
			tagged_actors
		);

		if (tagged_actors.Num() > 0) {
			this->pcg_volume_actor = tagged_actors[0];
			if (GEngine)
				GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green,
					TEXT("[Drone] PCG Volume auto-assigned"));
		}
		else {
			if (GEngine)
				GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
					TEXT("[Drone] ERROR: No actor with tag PCGVolume found"));
			UE_LOG(LogTemp, Error, TEXT("[DRONE] PCG Volume actor couldnt be auto detected in code."));
		}
	}
}

// Called every frame
void ADroneParent::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	if (!this->patrol_active) {
		if (!is_x_moving) {
			this->x_velocity = FMath::FInterpTo(
				this->x_velocity,
				0.0f,
				DeltaTime,
				this->damping_fac
			);
		}

		if (!is_y_moving) {
			this->y_velocity = FMath::FInterpTo(
				this->y_velocity,
				0.0f,
				DeltaTime,
				this->damping_fac
			);
		}
	}

	// Tick patrol movement
	this->TickPatrol(DeltaTime);

	// other logical function calls
	this->UpdateVolumeAwareness();

	// set new location with integrated velocity.
	FVector loc = GetActorLocation();
	loc.X += this->x_velocity * DeltaTime;
	loc.Y += this->y_velocity * DeltaTime;
	loc.Z = this->fixed_elevation;
	SetActorLocation(loc);

	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(
			0,
			0.0f,
			FColor::Green,
			FString::Printf(
				TEXT("X: %.1f  Y: %.1f  VelX: %.1f  VelY: %.1f"),
				loc.X,
				loc.Y,
				this->x_velocity,
				this->y_velocity
			)
		);

	}

	// Auto-save every N ticks
	this->tick_counter++;
	if (this->tick_counter >= this->capture_every_n_steps) {
		this->tick_counter = 0;
		this->SaveCurrentFrame();
	}

	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(0, 0.0f, FColor::Green,
			FString::Printf(TEXT("X: %.1f  Y: %.1f  VelX: %.1f  VelY: %.1f | Frame: %d"),
				loc.X, loc.Y, this->x_velocity, this->y_velocity,
				this->frame_save_index));
	}
}

// Called to bind functionality to input
void ADroneParent::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) {
	Super::SetupPlayerInputComponent(PlayerInputComponent);
	PlayerInputComponent->BindKey(EKeys::W, IE_Pressed, this, &ADroneParent::MoveAhead);
	PlayerInputComponent->BindKey(EKeys::W, IE_Released, this, &ADroneParent::StopX);

	PlayerInputComponent->BindKey(EKeys::S, IE_Pressed, this, &ADroneParent::MoveBack);
	PlayerInputComponent->BindKey(EKeys::S, IE_Released, this, &ADroneParent::StopX);

	PlayerInputComponent->BindKey(EKeys::A, IE_Pressed, this, &ADroneParent::MoveLeft);
	PlayerInputComponent->BindKey(EKeys::A, IE_Released, this, &ADroneParent::StopY);

	PlayerInputComponent->BindKey(EKeys::D, IE_Pressed, this, &ADroneParent::MoveRight);
	PlayerInputComponent->BindKey(EKeys::D, IE_Released, this, &ADroneParent::StopY);
}


// calculating PCG volumetric bounds and calculating relative position
// and also norm-ing it for RL calculations downstream.
// and setting a bool if inside 2D area cast by the volume on 2D ground plane.
void ADroneParent::UpdateVolumeAwareness() {
	if (!pcg_volume_actor) {
		if (GEngine) {
			GEngine->AddOnScreenDebugMessage(
				1,
				0.0f,
				FColor::Red,
				FString::Printf(
					TEXT("[ERROR] : UpdateVolumeAwareness - No PCG Volume Detected!")
				)
			);
		}
		return;
	}

	FVector vol_origin;
	FVector vol_extent;
	this->pcg_volume_actor->GetActorBounds(false, vol_origin, vol_extent);

	FVector drone_loc = GetActorLocation();
	FVector vol_min = vol_origin - vol_extent;
	FVector vol_max = vol_origin + vol_extent;

	this->is_inside_pcg =
		(drone_loc.X >= vol_min.X) &&
		(drone_loc.X <= vol_max.X) &&
		(drone_loc.Y >= vol_min.Y) &&
		(drone_loc.Y <= vol_max.Y);

	this->normalized_pcg_position.X = FMath::Clamp(
		(drone_loc.X - vol_min.X) / (vol_max.X - vol_min.X),
		0.0f,
		1.0f
	);
	this->normalized_pcg_position.Y = FMath::Clamp(
		(drone_loc.Y - vol_min.Y) / (vol_max.Y - vol_min.Y),
		0.0f,
		1.0f
	);

	// zero mask for Vol-OOB areas
	if (this->rt_volumetric_mask_material) {
		FVector2D screen_min, screen_max;
		
		float dist_to_min_x = drone_loc.X - vol_min.X;
		float dist_to_max_x = vol_max.X - drone_loc.X;
		float dist_to_min_y = drone_loc.Y - vol_min.Y;
		float dist_to_max_y = vol_max.Y - drone_loc.Y;

		float total_x = vol_max.X - vol_min.X;
		float total_y = vol_max.Y - vol_min.Y;

		screen_min.X = 0.5f - (dist_to_min_x / total_x);
		screen_max.X = 0.5f + (dist_to_max_x / total_x);
		screen_min.Y = 0.5f - (dist_to_min_y / total_y);
		screen_max.Y = 0.5f + (dist_to_max_y / total_y);

		this->rt_volumetric_mask_material->SetScalarParameterValue("BoundsMinU", screen_min.X);
		this->rt_volumetric_mask_material->SetScalarParameterValue("BoundsMinV", screen_min.Y);
		this->rt_volumetric_mask_material->SetScalarParameterValue("BoundsSizeU", screen_max.X - screen_min.X);
		this->rt_volumetric_mask_material->SetScalarParameterValue("BoundsSizeV", screen_max.Y - screen_min.Y);
	}

	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(
			1,
			0.0f,
			FColor::Yellow,
			FString::Printf(
				TEXT("NormPos: (%.2f, %.2f) Inside: %s"),
				this->normalized_pcg_position.X,
				this->normalized_pcg_position.Y,
				this->is_inside_pcg ? TEXT("YES") : TEXT("NO")
			)
		);
	}
}



void ADroneParent::UpdatePCGFrame() {
	if (!this->pcg_volume_actor) {
		this->pcg_frame_valid = false;
		return;
	}

	FVector origin, extent;
	this->pcg_volume_actor->GetActorBounds(false, origin, extent);

	// Cache bounds
	this->pcg_vol_min = FVector2D(origin.X - extent.X, origin.Y - extent.Y);
	this->pcg_vol_max = FVector2D(origin.X + extent.X, origin.Y + extent.Y);
	this->pcg_vol_extent = FVector2D(extent.X, extent.Y);
	this->pcg_volume_centre = FVector2D(origin.X, origin.Y);
	this->pcg_volume_size = FVector2D(extent.X * 2.0f, extent.Y * 2.0f);
	this->pcg_frame_valid = true;

	// Drone position transforms
	FVector2D drone_xy = FVector2D(
		GetActorLocation().X,
		GetActorLocation().Y
	);

	// Local: metres from volume centre (UE units)
	this->pcg_local_position = drone_xy - this->pcg_volume_centre;

	// Normalized: [-1, 1] where (0,0) = volume centre
	this->pcg_normalized_position = FVector2D(
		FMath::Clamp(
			this->pcg_local_position.X / this->pcg_vol_extent.X,
			-1.0f, 1.0f
		),
		FMath::Clamp(
			this->pcg_local_position.Y / this->pcg_vol_extent.Y,
			-1.0f, 1.0f
		)
	);

	// Unorm: [0, 1] where (0,0) = volume min corner (top-left)
	this->pcg_unorm_position = FVector2D(
		FMath::Clamp(
			(drone_xy.X - this->pcg_vol_min.X) / this->pcg_volume_size.X,
			0.0f, 1.0f
		),
		FMath::Clamp(
			(drone_xy.Y - this->pcg_vol_min.Y) / this->pcg_volume_size.Y,
			0.0f, 1.0f
		)
	);

	// Velocity in PCG frame
	// Raw velocity (UU/s) — already in world frame which aligns with PCG
	this->pcg_velocity_local = FVector2D(
		this->x_velocity,
		this->y_velocity
	);

	// Velocity normalized by volume size
	// gives RL agent a scale-invariant velocity signal
	this->pcg_velocity_normalized = FVector2D (
		this->pcg_volume_size.X > 0.0f ? this->x_velocity / this->pcg_volume_size.X : 0.0f,
		this->pcg_volume_size.Y > 0.0f ? this->y_velocity / this->pcg_volume_size.Y : 0.0f
	);
}



FVector2D ADroneParent::WorldToPCGLocal(FVector2D world_xy) const {
	if (!this->pcg_frame_valid) return FVector2D::ZeroVector;
	return world_xy - this->pcg_volume_centre;
}

FVector2D ADroneParent::WorldToPCGNormalized(FVector2D world_xy) const {
	if (!this->pcg_frame_valid) return FVector2D::ZeroVector;

	FVector2D local = world_xy - this->pcg_volume_centre;
	return FVector2D(
		FMath::Clamp(local.X / this->pcg_vol_extent.X, -1.0f, 1.0f),
		FMath::Clamp(local.Y / this->pcg_vol_extent.Y, -1.0f, 1.0f)
	);
}

FVector2D ADroneParent::WorldToPCGUnorm(FVector2D world_xy) const {
	if (!this->pcg_frame_valid) return FVector2D::ZeroVector;

	return FVector2D(
		FMath::Clamp(
			(world_xy.X - this->pcg_vol_min.X) / this->pcg_volume_size.X,
			0.0f, 1.0f),
		FMath::Clamp(
			(world_xy.Y - this->pcg_vol_min.Y) / this->pcg_volume_size.Y,
			0.0f, 1.0f)
	);
}

FVector2D ADroneParent::PCGNormalizedToWorld(FVector2D pcg_norm) const {
	if (!this->pcg_frame_valid) return FVector2D::ZeroVector;

	return FVector2D(
		this->pcg_volume_centre.X + pcg_norm.X * this->pcg_vol_extent.X,
		this->pcg_volume_centre.Y + pcg_norm.Y * this->pcg_vol_extent.Y
	);
}

FVector2D ADroneParent::PCGUnormToWorld(FVector2D pcg_unorm) const {
	if (!this->pcg_frame_valid) return FVector2D::ZeroVector;

	return FVector2D(
		this->pcg_vol_min.X + pcg_unorm.X * this->pcg_volume_size.X,
		this->pcg_vol_min.Y + pcg_unorm.Y * this->pcg_volume_size.Y
	);
}

FString ADroneParent::GetRLObservationJSON() const {
	TSharedPtr<FJsonObject> root = MakeShareable(new FJsonObject);

	// ── Position ─────────────────────────────────────────────────
	TSharedPtr<FJsonObject> pos = MakeShareable(new FJsonObject);

	// PCG-local (UE units, origin = volume centre)
	pos->SetNumberField(TEXT("local_x"), this->pcg_local_position.X);
	pos->SetNumberField(TEXT("local_y"), this->pcg_local_position.Y);

	// Normalized [-1, 1]
	pos->SetNumberField(TEXT("norm_x"), this->pcg_normalized_position.X);
	pos->SetNumberField(TEXT("norm_y"), this->pcg_normalized_position.Y);

	// Unorm [0, 1]
	pos->SetNumberField(TEXT("unorm_x"), this->pcg_unorm_position.X);
	pos->SetNumberField(TEXT("unorm_y"), this->pcg_unorm_position.Y);

	// World position (for logging/debugging only, not fed to RL)
	pos->SetNumberField(TEXT("world_x"), GetActorLocation().X);
	pos->SetNumberField(TEXT("world_y"), GetActorLocation().Y);
	pos->SetNumberField(TEXT("world_z"), GetActorLocation().Z);

	root->SetObjectField(TEXT("position"), pos);

	// ── Velocity ─────────────────────────────────────────────────
	TSharedPtr<FJsonObject> vel = MakeShareable(new FJsonObject);
	vel->SetNumberField(TEXT("local_vx"), this->pcg_velocity_local.X);
	vel->SetNumberField(TEXT("local_vy"), this->pcg_velocity_local.Y);
	vel->SetNumberField(TEXT("norm_vx"), this->pcg_velocity_normalized.X);
	vel->SetNumberField(TEXT("norm_vy"), this->pcg_velocity_normalized.Y);
	root->SetObjectField(TEXT("velocity"), vel);

	// ── Volume info ───────────────────────────────────────────────
	TSharedPtr<FJsonObject> vol = MakeShareable(new FJsonObject);
	vol->SetNumberField(TEXT("centre_x"), this->pcg_volume_centre.X);
	vol->SetNumberField(TEXT("centre_y"), this->pcg_volume_centre.Y);
	vol->SetNumberField(TEXT("size_x"), this->pcg_volume_size.X);
	vol->SetNumberField(TEXT("size_y"), this->pcg_volume_size.Y);
	vol->SetBoolField(TEXT("inside"), this->is_inside_pcg);
	root->SetObjectField(TEXT("volume"), vol);

	// ── Fire info ─────────────────────────────────────────────────
	TArray<TSharedPtr<FJsonValue>> fire_arr;

	for (AStaticMeshActor* marker : this->fire_stencil_markers) {
		if (!marker) continue;

		FVector2D fire_world = FVector2D(
			marker->GetActorLocation().X,
			marker->GetActorLocation().Y
		);

		FVector2D fire_norm = WorldToPCGNormalized(fire_world);
		FVector2D fire_unorm = WorldToPCGUnorm(fire_world);

		TSharedPtr<FJsonObject> fire_obj = MakeShareable(new FJsonObject);
		fire_obj->SetNumberField(TEXT("norm_x"), fire_norm.X);
		fire_obj->SetNumberField(TEXT("norm_y"), fire_norm.Y);
		fire_obj->SetNumberField(TEXT("unorm_x"), fire_unorm.X);
		fire_obj->SetNumberField(TEXT("unorm_y"), fire_unorm.Y);
		fire_arr.Add(MakeShareable(new FJsonValueObject(fire_obj)));
	}

	root->SetArrayField(TEXT("fire_positions"), fire_arr);

	// Episode info 
	root->SetStringField(TEXT("session_id"), this->session_id);
	root->SetNumberField(TEXT("frame_id"), this->frame_save_index);
	root->SetNumberField(TEXT("fire_count"), this->fire_stencil_markers.Num());

	FString out;
	TSharedRef<TJsonWriter<>> writer = TJsonWriterFactory<>::Create(&out);
	FJsonSerializer::Serialize(root.ToSharedRef(), writer);
	return out;
}



void ADroneParent::MoveAhead() {
	this->x_velocity = this->speed; 
	this->is_x_moving = true;
}

void ADroneParent::MoveBack() {
	this->x_velocity = -1 * this->speed;
	this->is_x_moving = true;
}

void ADroneParent::MoveLeft() {
	this->y_velocity = -1 * this->speed;
	this->is_y_moving = true;
}

void ADroneParent::MoveRight() {
	this->y_velocity = this->speed;
	this->is_y_moving = true;
}

void ADroneParent::StopX() {
	this->is_x_moving = false;
}

void ADroneParent::StopY() {
	this->is_y_moving = false;
}

void ADroneParent::SetPCGVolume(AActor* pcg_actor) {
	this->pcg_volume_actor = pcg_actor;
}
