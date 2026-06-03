// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "UObject/NoExportTypes.h"
#include "ConfigManager.generated.h"

USTRUCT(BlueprintType)
struct FDroneConfig
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) float fixed_elevation = 2800.0f;
    UPROPERTY(BlueprintReadOnly) float speed = 500.0f;
    UPROPERTY(BlueprintReadOnly) float damping_factor = 1.0f;
    UPROPERTY(BlueprintReadOnly) int32 capture_every_n_steps = 30;
    UPROPERTY(BlueprintReadOnly) float patrol_step_size = 1000.0f;
    UPROPERTY(BlueprintReadOnly) float waypoint_threshold = 100.0f;
    UPROPERTY(BlueprintReadOnly) float patrol_speed = 300.0f;
    UPROPERTY(BlueprintReadOnly) bool  randomize_pcg_seed = true;
    UPROPERTY(BlueprintReadOnly) bool  inference_enabled = false;
    UPROPERTY(BlueprintReadOnly) bool  use_websocket = false;
    UPROPERTY(BlueprintReadOnly) FString inference_server_url = TEXT("http://localhost:8000");
    UPROPERTY(BlueprintReadOnly) FString websocket_server_url = TEXT("ws://localhost:8000/ws/infer/");
    UPROPERTY(BlueprintReadOnly) bool  is_networking_active = true;
};

USTRUCT(BlueprintType)
struct FFireConfig
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) int32 num_fire_seeds = 5;
    UPROPERTY(BlueprintReadOnly) float spread_radius = 400.0f;
    UPROPERTY(BlueprintReadOnly) float emitter_scale_min = 0.5f;
    UPROPERTY(BlueprintReadOnly) float emitter_scale_max = 2.0f;
    UPROPERTY(BlueprintReadOnly) float marker_scale = 300.0f;
};

USTRUCT(BlueprintType)
struct FCaptureConfig
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) FString save_directory = TEXT("C:/WildfireDataset/");
    UPROPERTY(BlueprintReadOnly) bool save_rgb = true;
    UPROPERTY(BlueprintReadOnly) bool save_color_mask = true;
    UPROPERTY(BlueprintReadOnly) bool save_binary_masks = true;
    UPROPERTY(BlueprintReadOnly) bool save_index_mask = true;
    UPROPERTY(BlueprintReadOnly) bool save_metadata = true;
    UPROPERTY(BlueprintReadOnly) int32 image_width = 512;
    UPROPERTY(BlueprintReadOnly) int32 image_height = 512;
};

USTRUCT(BlueprintType)
struct FServerConfig
{
    GENERATED_BODY()

    UPROPERTY(BlueprintReadOnly) FString host = TEXT("localhost");
    UPROPERTY(BlueprintReadOnly) int32   port = 8080;
    UPROPERTY(BlueprintReadOnly) FString websocket_path = TEXT("/");
    UPROPERTY(BlueprintReadOnly) FString http_infer_path = TEXT("/infer/");
    UPROPERTY(BlueprintReadOnly) FString http_health_path = TEXT("/health/");
};

UCLASS(BlueprintType)
class PCGFOREST2_API UConfigManager : public UObject
{
    GENERATED_BODY()

public:
    // Singleton access
    static UConfigManager* Get();

    // Reload config from disk at any time
    UFUNCTION(BlueprintCallable, Category = "Config")
    void LoadConfig();

    // Save current config back to disk
    UFUNCTION(BlueprintCallable, Category = "Config")
    void SaveConfig();

    // Print all loaded values to screen
    UFUNCTION(BlueprintCallable, Category = "Config")
    void DebugPrintAll() const;

    // Config structs — read from anywhere
    UPROPERTY(BlueprintReadOnly, Category = "Config")
    FDroneConfig drone;

    UPROPERTY(BlueprintReadOnly, Category = "Config")
    FFireConfig fire;

    UPROPERTY(BlueprintReadOnly, Category = "Config")
    FCaptureConfig capture;

    UPROPERTY(BlueprintReadOnly, Category = "Config")
    FServerConfig server;

    // Helper to build full server URLs
    FString GetHttpInferUrl()  const;
    FString GetHttpHealthUrl() const;
    FString GetWebSocketUrl()  const;

    bool IsNetworkingActive();

private:
    static UConfigManager* instance;

    FString GetConfigFilePath() const;

    // INI helpers
    void ReadFloat(const FString& section, const FString& key, float& out) const;
    void ReadInt(const FString& section, const FString& key, int32& out) const;
    void ReadBool(const FString& section, const FString& key, bool& out) const;
    void ReadString(const FString& section, const FString& key, FString& out) const;

    void WriteFloat(const FString& section, const FString& key, float   val);
    void WriteInt(const FString& section, const FString& key, int32   val);
    void WriteBool(const FString& section, const FString& key, bool    val);
    void WriteString(const FString& section, const FString& key, const FString& val);
};