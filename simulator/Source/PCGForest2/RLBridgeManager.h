// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "WebSocketsModule.h"
#include "IWebSocket.h"
#include "Engine/TextureRenderTarget2D.h"
#include "RLBridgeManager.generated.h"

/**
 * 
 */
 // Action received from Python RL
USTRUCT(BlueprintType)
struct FRLAction {
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) int32 dx = 1; // 0=neg, 1=stay, 2=pos
    UPROPERTY(BlueprintReadOnly) int32 dy = 1;
    UPROPERTY(BlueprintReadOnly) int32 step_id = 0;
};

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(
    FOnRLActionReceived, FRLAction, action);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(
    FOnRLResetRequested);
DECLARE_DYNAMIC_MULTICAST_DELEGATE(
    FOnRLConnected);

UCLASS(BlueprintType)
class PCGFOREST2_API URLBridgeManager : public UObject
{
    GENERATED_BODY()

public:
    static URLBridgeManager* Get();

    // Connect to Python RL server
    UFUNCTION(BlueprintCallable, Category = "RL")
    void Connect(const FString& url);

    UFUNCTION(BlueprintCallable, Category = "RL")
    void Disconnect();

    UFUNCTION(BlueprintCallable, Category = "RL")
    bool IsConnected() const { return connected; }

    // Send observation to Python
    void SendObservation(
        UTextureRenderTarget2D* seg_rt,
        FVector2D norm_pos,
        FVector2D norm_vel,
        FVector2D vol_size_uu,
        float     elevation,
        float     fov_degrees,
        TArray<FVector2D> fire_norm_positions,
        bool      inside_pcg,
        int32     step_id,
        bool      episode_done
    );

    // Send episode reset confirmation
    void SendResetConfirm(
        FVector2D vol_size_uu,
        float     elevation,
        float     fov_degrees,
        FVector2D drone_start_norm
    );

    // Send episode done notification
    void SendEpisodeDone(int32 total_steps, float total_reward);

    // Events
    UPROPERTY(BlueprintAssignable)
    FOnRLActionReceived  OnActionReceived;

    UPROPERTY(BlueprintAssignable)
    FOnRLResetRequested  OnResetRequested;

    UPROPERTY(BlueprintAssignable)
    FOnRLConnected       OnRLConnected;

    // Step gating — UE5 waits for action before simulating
    bool waiting_for_action = false;
    FRLAction pending_action;

private:
    static URLBridgeManager* instance;
    TSharedPtr<IWebSocket> socket;
    bool connected = false;

    void HandleConnected();
    void HandleError(const FString& error);
    void HandleClosed(int32 code, const FString& reason, bool clean);
    void HandleMessage(const FString& message);

    void ParseAction(TSharedPtr<FJsonObject> json);
    void ParseReset(TSharedPtr<FJsonObject> json);

    FString EncodeRT(UTextureRenderTarget2D* rt);
};
