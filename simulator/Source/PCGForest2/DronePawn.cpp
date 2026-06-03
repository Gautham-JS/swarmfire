// Fill out your copyright notice in the Description page of Project Settings.


#include "DronePawn.h"
#include "Camera/CameraComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/InputComponent.h"


// Sets default values
ADronePawn::ADronePawn()
{
 	// Set this pawn to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	drone_mesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("DroneMesh"));
	drone_mesh->SetRelativeRotation(FRotator(0.0, 0.0, -90.0f));
	drone_mesh->SetRelativeScale3D(FVector(3.0, 3.0, 3.0));
	RootComponent = drone_mesh;

	drone_mesh->SetEnableGravity(false);
	drone_mesh->SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);

	camera_front = CreateDefaultSubobject<UCameraComponent>(TEXT("CameraFront"));
	camera_front->SetupAttachment(drone_mesh);
	camera_front->SetRelativeLocation(FVector(0.0, 0.0, 0.0));
	camera_front->SetRelativeRotation(FRotator(0.0, 0.0, 0.0));
	camera_front->SetRelativeScale3D(FVector(0.2, 0.2, 0.2));

	camera_down = CreateDefaultSubobject<UCameraComponent>(TEXT("CameraDown"));
	camera_down->SetupAttachment(drone_mesh);
	camera_down->SetRelativeScale3D(FVector(0.2, 0.2, 0.2));
	camera_down->SetRelativeLocation(FVector(0.0, 0.0, -220.0));
	camera_down->SetRelativeRotation(FRotator(0.0, -90.0, 0.0));

	AutoPossessPlayer = EAutoReceiveInput::Player0;
}

// Called when the game starts or when spawned
void ADronePawn::BeginPlay()
{
	Super::BeginPlay();

	// setting the fixed elevation here
	FVector start_loc = GetActorLocation();
	start_loc.Z = this->fixed_elevation;
	SetActorLocation(start_loc);
}

// Called every frame
void ADronePawn::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	if (!is_moving_x) {
		this->x_velocity = FMath::FInterpTo(
			this->x_velocity, 0.0f, DeltaTime, this->damping_fac
		);
	}

	if (!is_moving_y) {
		this->y_velocity = FMath::FInterpTo(
			this->y_velocity, 0.0f, DeltaTime, this->damping_fac
		);
	}

	// core velocity controller logic
	FVector loc = GetActorLocation();

	loc.X += this->x_velocity * DeltaTime;
	loc.Y += this->y_velocity * DeltaTime;
	loc.Z = this->fixed_elevation;

	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(
			0, // to override same msg
			0.0f, // idk why would you want a separate delay but I digress
			FColor::Green,
			FString::Printf(TEXT("X: %.1f  Y: %.1f"), loc.X, loc.Y)
		);
	}


}

// Called to bind functionality to input
void ADronePawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	PlayerInputComponent->BindAction(
		"MoveForward",
		IE_Pressed,
		this,
		&ADronePawn::MoveAhead
	);

	PlayerInputComponent->BindAction(
		"MoveForward",
		IE_Released,
		this,
		&ADronePawn::StopX
	);

	PlayerInputComponent->BindAction(
		"MoveBehind",
		IE_Pressed,
		this,
		&ADronePawn::MoveBack
	);


	PlayerInputComponent->BindAction(
		"MoveBehind",
		IE_Released,
		this,
		&ADronePawn::StopX
	);
}



void ADronePawn::MoveAhead() {
	this->x_velocity = this->speed;
	this->is_moving_x = true;
}

void ADronePawn::MoveBack() {
	this->x_velocity = -1 * this->speed;
	this->is_moving_x = true;
}

void ADronePawn::MoveLeft() {
	this->y_velocity = -1 * this->speed;
	this->is_moving_y = true;
}

void ADronePawn::MoveRight() {
	this->y_velocity = this->speed;
	this->is_moving_y = true;
}

void ADronePawn::StopX() {
	this->is_moving_x = false;
}

void ADronePawn::StopY() {
	this->is_moving_y = true;
}




