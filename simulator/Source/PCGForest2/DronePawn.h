// Fill out your copyright notice in the Description page of Project Settings.

#pragma once


#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "Camera/CameraComponent.h"
#include "Components/StaticMeshComponent.h"
#include "Components/InputComponent.h"
#include "Engine/Engine.h"
#include "GameFramework/PlayerController.h"


#include "DronePawn.generated.h"



UCLASS()
class PCGFOREST2_API ADronePawn : public APawn
{
	GENERATED_BODY()

public:
	// Sets default values for this pawn's properties
	ADronePawn();

protected:

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UStaticMeshComponent* drone_mesh;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UCameraComponent* camera_front;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UCameraComponent* camera_down;




	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;


private:
	float x_velocity = 0.0f;
	float y_velocity = 0.0f;

	float damping_fac = 5.0f;
	float speed = 500.0f;
	float fixed_elevation = 2800.0f;

	bool is_moving_x = false;
	bool is_moving_y = false;

	void MoveLeft();
	void MoveRight();
	void MoveAhead();
	void MoveBack();

	void StopX();
	void StopY();


};
