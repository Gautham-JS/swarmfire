// Fill out your copyright notice in the Description page of Project Settings.


#include "EnvHandler.h"
#include "Engine/Engine.h"
#include "EngineUtils.h"
#include "Engine/StaticMeshActor.h"
#include "Components/StaticMeshComponent.h"
#include "Components/HierarchicalInstancedStaticMeshComponent.h"
#include "UObject/UObjectIterator.h"

#include "Kismet/GameplayStatics.h"

// Sets default values
AEnvHandler::AEnvHandler(){
 	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

}

// Called when the game starts or when spawned
void AEnvHandler::BeginPlay() {
	Super::BeginPlay();
	
	this->FindPCGComponent();
	this->SetupWebsocket();
	this->ResetEnv();
	
}

void AEnvHandler::EndPlay(EEndPlayReason::Type reason) {
	this->CloseEnv();
	Super::EndPlay(reason);
}

void AEnvHandler::CloseEnv() {

}

TSharedPtr<FJsonObject> AEnvHandler::ParseJSON(const FString &json) {
	return nullptr;
}

void AEnvHandler::HandleWebsocketMessage(const FString& msg) {

}

// Called every frame
void AEnvHandler::Tick(float DeltaTime) {
	Super::Tick(DeltaTime);
	//this->drone_instance->Tick(DeltaTime);
}


/*
* - Mirroring RL framework func to reset env to defaults
*/
void AEnvHandler::ResetEnv() {
	environment_ready = false;
	DestroyDrone();
	RegeneratePCG();
}

/* 
* ---------------
* - Blows up the drone, AKA cleans up the ptrs
* --------------- 
*/ 
void AEnvHandler::DestroyDrone() {
	if (this->drone_instance) { this->drone_instance->Destroy(); this->drone_instance = nullptr; }
}



/*
* ---------------
* - Finds existing PCG component from the level heirarchy
* ---------------
*/
void AEnvHandler::FindPCGComponent() {
	UWorld* world = GetWorld();
	if (!world) return;

	for (TActorIterator<AActor> it(world); it; ++it) {
		UPCGComponent* comp = it->FindComponentByClass<UPCGComponent>();
		if (comp) {
			this->pcg_component = comp;
			UE_LOG(LogTemp, Warning, TEXT("[ENV] PCG component found"));
			return;
		}
	}
	UE_LOG(LogTemp, Error, TEXT("[ENV] Failed to find PCG component"));
}


/*
* - Sets up transport layer (WebSockets interface)
*/
void AEnvHandler::SetupWebsocket() {
	if (!this->auto_connect_ws) { return; }

	WebsocketManager::Clean();
	WebsocketManager* ws = WebsocketManager::Get();
	ws->Connect(this->rl_server_url);
	UE_LOG(LogTemp, Warning, TEXT("[ENV] Websocket connected"));
}


/*
* - Spawns in the drone blueprint to the level
* - Assigns the drone_instance ptr
*/ 
void AEnvHandler::SpawnDrone() {
	if (!this->drone_class) {
		UE_LOG(LogTemp, Error, TEXT("[ENV] drone_class is null"));
		return;
	}

	FActorSpawnParameters params;
	params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AdjustIfPossibleButAlwaysSpawn;
	this->drone_instance = GetWorld()->SpawnActor<ADroneParent>(
		this->drone_class,
		this->drone_spawn_location,
		this->drone_spawn_rotation,
		params
	);
	if (!this->drone_instance) {
		UE_LOG(LogTemp, Error, TEXT("[ENV] Failed to spawn drone"));
		return;
	}
	// TODO: Add mode switching, Patrol default rn
	
	UE_LOG(LogTemp, Warning, TEXT("[ENV] Drone spawned with name %s"), *this->drone_instance->GetName());

}


void AEnvHandler::RegeneratePCG() {
	if (!this->pcg_component) {
		UE_LOG(LogTemp, Error, TEXT("[ENV] No PCG component"));
		return;
	}
	this->pcg_component->OnPCGGraphGeneratedDelegate.AddUObject(
		this,
		&AEnvHandler::OnPCGGenerated
	);

	int32 seed = FMath::Rand();
	this->pcg_component->Seed = seed;

	UE_LOG(LogTemp, Warning, TEXT("[ENV] Regenerating PCG seed=%d"), seed);
	// TODO: Get rid of these calls / wrap then in definitions for Editor mode
	this->pcg_component->DirtyGenerated();
	this->pcg_component->Generate();
}



void AEnvHandler::OnPCGGenerated(UPCGComponent* comp) {
	comp->OnPCGGraphGeneratedDelegate.RemoveAll(this);
	UE_LOG(LogTemp, Warning, TEXT("[ENV] PCG generation complete"));

	SpawnDrone();
	if (!this->drone_instance) return;
	this->drone_instance->SetPCGVolume(this->pcg_component->GetOwner());
	this->drone_instance->SetTreeMeshStencilIDs();

	/*if (environment_mode == EEnvironmentMode::Patrol) {
		this->drone->StartGridPatrol();
	}*/
	this->drone_instance->StartGridPatrol();
	this->environment_ready = true;
	SendObservation(false);
}

void AEnvHandler::StepEnv(const FString& action_json) {

	if (!this->drone_instance) return;

	TSharedPtr<FJsonObject> root = ParseJSON(action_json);
	if (!root.IsValid()) return;

	int32 dx = root->GetIntegerField(TEXT("dx"));
	int32 dy = root->GetIntegerField(TEXT("dy"));

	if (dx == 0)		this->drone_instance->StopX();
	else if (dx > 0)	this->drone_instance->MoveAhead();
	else				this->drone_instance->MoveBack();

	if (dy == 0)		this->drone_instance->StopY();
	else if (dy > 0)	this->drone_instance->MoveRight();
	else				this->drone_instance->MoveLeft();

	SendObservation(false);
}


void AEnvHandler::SendObservation(bool done) {
	if (!this->drone_instance) return;

	FString obs = this->drone_instance->GetRLObservationJSON();

	TSharedPtr<FJsonObject> root = MakeShareable(new FJsonObject);
	root->SetStringField(TEXT("type"), TEXT("obs"));
	root->SetBoolField(TEXT("done"), done);

	TSharedPtr<FJsonObject> obs_json;
	TSharedRef<TJsonReader<>> reader = TJsonReaderFactory<>::Create(obs);
	FJsonSerializer::Deserialize(reader, obs_json);

	root->SetObjectField(TEXT("observation"), obs_json);

	FString out;
	TSharedRef<TJsonWriter<>> writer = TJsonWriterFactory<>::Create(&out);
	FJsonSerializer::Serialize(root.ToSharedRef(), writer);

	// TODO: Replace with binary SendRawMessage func for speed.
	WebsocketManager::Get()->SendJsonMessage("obs", out);
}


void AEnvHandler::SetTreeMeshStencilIDs() {
	UE_LOG(LogTemp, Warning, TEXT("[ENV] SetTreeMeshStencilIDs -- enter"));

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
	UE_LOG(LogTemp, Warning, TEXT("[ENV] SetTreeMeshStencilIDs -- exit | S1 : %d | S2 : %d"), stencil_1_count, stencil_2_count);
}

