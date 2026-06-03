// Fill out your copyright notice in the Description page of Project Settings.


#include "ConfigManager.h"
#include "Misc/ConfigCacheIni.h"
#include "Misc/Paths.h"
#include "HAL/FileManager.h"

UConfigManager* UConfigManager::instance = nullptr;

UConfigManager* UConfigManager::Get() {
    if (!instance || !instance->IsValidLowLevel()) {
        instance = NewObject<UConfigManager>();
        instance->AddToRoot(); // prevent GC
        instance->LoadConfig();
    }
    return instance;
}

FString UConfigManager::GetConfigFilePath() const {
    return FPaths::ProjectConfigDir() + TEXT("WildfireCapture.ini");
}

void UConfigManager::LoadConfig() {
    FString config_path = GetConfigFilePath();

    if (!FPaths::FileExists(config_path)) {
        if (GEngine)
            GEngine->AddOnScreenDebugMessage(-1, 10.f, FColor::Yellow,
                FString::Printf(TEXT("[Config] File not found: %s — using defaults"),
                    *config_path));
        return;
    }

    // Force reload from disk
    GConfig->LoadFile(config_path);

    // --- Drone ---
    ReadFloat(TEXT("DroneConfig"), TEXT("FixedElevation"), drone.fixed_elevation);
    ReadFloat(TEXT("DroneConfig"), TEXT("Speed"), drone.speed);
    ReadFloat(TEXT("DroneConfig"), TEXT("DampingFactor"), drone.damping_factor);
    ReadInt(TEXT("DroneConfig"), TEXT("CaptureEveryNSteps"), drone.capture_every_n_steps);
    ReadFloat(TEXT("DroneConfig"), TEXT("PatrolStepSize"), drone.patrol_step_size);
    ReadFloat(TEXT("DroneConfig"), TEXT("WaypointReachThreshold"), drone.waypoint_threshold);
    ReadFloat(TEXT("DroneConfig"), TEXT("PatrolSpeed"), drone.patrol_speed);
    ReadBool(TEXT("DroneConfig"), TEXT("RandomizePCGSeed"), drone.randomize_pcg_seed);
    ReadBool(TEXT("DroneConfig"), TEXT("InferenceEnabled"), drone.inference_enabled);
    ReadBool(TEXT("DroneConfig"), TEXT("UseWebSocket"), drone.use_websocket);
    ReadString(TEXT("DroneConfig"), TEXT("InferenceServerUrl"), drone.inference_server_url);
    ReadString(TEXT("DroneConfig"), TEXT("WebSocketServerUrl"), drone.websocket_server_url);
    ReadBool(TEXT("DroneConfig"), TEXT("IsNetworkingActive"), drone.is_networking_active);

    // --- Fire ---
    ReadInt(TEXT("FireConfig"), TEXT("NumFireSeeds"), fire.num_fire_seeds);
    ReadFloat(TEXT("FireConfig"), TEXT("SpreadRadius"), fire.spread_radius);
    ReadFloat(TEXT("FireConfig"), TEXT("EmitterScaleMin"), fire.emitter_scale_min);
    ReadFloat(TEXT("FireConfig"), TEXT("EmitterScaleMax"), fire.emitter_scale_max);
    ReadFloat(TEXT("FireConfig"), TEXT("MarkerScale"), fire.marker_scale);

    // --- Capture ---
    ReadString(TEXT("CaptureConfig"), TEXT("SaveDirectory"), capture.save_directory);
    ReadBool(TEXT("CaptureConfig"), TEXT("SaveRGB"), capture.save_rgb);
    ReadBool(TEXT("CaptureConfig"), TEXT("SaveColorMask"), capture.save_color_mask);
    ReadBool(TEXT("CaptureConfig"), TEXT("SaveBinaryMasks"), capture.save_binary_masks);
    ReadBool(TEXT("CaptureConfig"), TEXT("SaveIndexMask"), capture.save_index_mask);
    ReadBool(TEXT("CaptureConfig"), TEXT("SaveMetadata"), capture.save_metadata);
    ReadInt(TEXT("CaptureConfig"), TEXT("ImageWidth"), capture.image_width);
    ReadInt(TEXT("CaptureConfig"), TEXT("ImageHeight"), capture.image_height);

    // --- Server ---
    ReadString(TEXT("ServerConfig"), TEXT("Host"), server.host);
    ReadInt(TEXT("ServerConfig"), TEXT("Port"), server.port);
    ReadString(TEXT("ServerConfig"), TEXT("WebSocketPath"), server.websocket_path);
    ReadString(TEXT("ServerConfig"), TEXT("HttpInferPath"), server.http_infer_path);
    ReadString(TEXT("ServerConfig"), TEXT("HttpHealthPath"), server.http_health_path);

    if (GEngine)
        GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green,
            FString::Printf(TEXT("[Config] Loaded from: %s"), *config_path));
}

void UConfigManager::SaveConfig() {
    FString config_path = GetConfigFilePath();

    // --- Drone ---
    WriteFloat(TEXT("DroneConfig"), TEXT("FixedElevation"), drone.fixed_elevation);
    WriteFloat(TEXT("DroneConfig"), TEXT("Speed"), drone.speed);
    WriteFloat(TEXT("DroneConfig"), TEXT("DampingFactor"), drone.damping_factor);
    WriteInt(TEXT("DroneConfig"), TEXT("CaptureEveryNSteps"), drone.capture_every_n_steps);
    WriteFloat(TEXT("DroneConfig"), TEXT("PatrolStepSize"), drone.patrol_step_size);
    WriteFloat(TEXT("DroneConfig"), TEXT("WaypointReachThreshold"), drone.waypoint_threshold);
    WriteFloat(TEXT("DroneConfig"), TEXT("PatrolSpeed"), drone.patrol_speed);
    WriteBool(TEXT("DroneConfig"), TEXT("RandomizePCGSeed"), drone.randomize_pcg_seed);
    WriteBool(TEXT("DroneConfig"), TEXT("InferenceEnabled"), drone.inference_enabled);
    WriteBool(TEXT("DroneConfig"), TEXT("UseWebSocket"), drone.use_websocket);
    WriteString(TEXT("DroneConfig"), TEXT("InferenceServerUrl"), drone.inference_server_url);
    WriteString(TEXT("DroneConfig"), TEXT("WebSocketServerUrl"), drone.websocket_server_url);
    WriteBool(TEXT("DroneConfig"), TEXT("IsNetworkingActive"), drone.is_networking_active);

    // --- Fire ---
    WriteInt(TEXT("FireConfig"), TEXT("NumFireSeeds"), fire.num_fire_seeds);
    WriteFloat(TEXT("FireConfig"), TEXT("SpreadRadius"), fire.spread_radius);
    WriteFloat(TEXT("FireConfig"), TEXT("EmitterScaleMin"), fire.emitter_scale_min);
    WriteFloat(TEXT("FireConfig"), TEXT("EmitterScaleMax"), fire.emitter_scale_max);
    WriteFloat(TEXT("FireConfig"), TEXT("MarkerScale"), fire.marker_scale);

    // --- Capture ---
    WriteString(TEXT("CaptureConfig"), TEXT("SaveDirectory"), capture.save_directory);
    WriteBool(TEXT("CaptureConfig"), TEXT("SaveRGB"), capture.save_rgb);
    WriteBool(TEXT("CaptureConfig"), TEXT("SaveColorMask"), capture.save_color_mask);
    WriteBool(TEXT("CaptureConfig"), TEXT("SaveBinaryMasks"), capture.save_binary_masks);
    WriteBool(TEXT("CaptureConfig"), TEXT("SaveIndexMask"), capture.save_index_mask);
    WriteBool(TEXT("CaptureConfig"), TEXT("SaveMetadata"), capture.save_metadata);
    WriteInt(TEXT("CaptureConfig"), TEXT("ImageWidth"), capture.image_width);
    WriteInt(TEXT("CaptureConfig"), TEXT("ImageHeight"), capture.image_height);

    // --- Server ---
    WriteString(TEXT("ServerConfig"), TEXT("Host"), server.host);
    WriteInt(TEXT("ServerConfig"), TEXT("Port"), server.port);
    WriteString(TEXT("ServerConfig"), TEXT("WebSocketPath"), server.websocket_path);
    WriteString(TEXT("ServerConfig"), TEXT("HttpInferPath"), server.http_infer_path);
    WriteString(TEXT("ServerConfig"), TEXT("HttpHealthPath"), server.http_health_path);

    GConfig->Flush(false, config_path);

    if (GEngine)
        GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green,
            TEXT("[Config] Saved to disk"));
}

void UConfigManager::DebugPrintAll() const {
    if (!GEngine) return;

    GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::Cyan,
        TEXT("=== CONFIG DUMP ==="));

    // Drone
    GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::White,
        FString::Printf(TEXT("[Drone] Elevation=%.0f Speed=%.0f Damping=%.2f"),
            drone.fixed_elevation, drone.speed, drone.damping_factor));
    GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::White,
        FString::Printf(TEXT("[Drone] CaptureEvery=%d PatrolStep=%.0f PatrolSpeed=%.0f"),
            drone.capture_every_n_steps, drone.patrol_step_size, drone.patrol_speed));
    GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::White,
        FString::Printf(TEXT("[Drone] RandomPCG=%s Inference=%s WS=%s"),
            drone.randomize_pcg_seed ? TEXT("Y") : TEXT("N"),
            drone.inference_enabled ? TEXT("Y") : TEXT("N"),
            drone.use_websocket ? TEXT("Y") : TEXT("N")));

    // Fire
    GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::Orange,
        FString::Printf(TEXT("[Fire] Seeds=%d Spread=%.0f Scale=%.1f-%.1f"),
            fire.num_fire_seeds, fire.spread_radius,
            fire.emitter_scale_min, fire.emitter_scale_max)
    );

    // Capture
    GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::Yellow, FString::Printf(TEXT("[Capture] Dir=%s Size=%dx%d"),
            *capture.save_directory,
            capture.image_width, capture.image_height)
    );

    // Server
    GEngine->AddOnScreenDebugMessage(-1, 15.f, FColor::Green, FString::Printf(TEXT("[Server] %s:%d"), *server.host, server.port));
}

// URL helpers
FString UConfigManager::GetHttpInferUrl() const {
    return FString::Printf(TEXT("http://%s:%d%s"), *server.host, server.port, *server.http_infer_path);
}

bool UConfigManager::IsNetworkingActive() {
    return drone.is_networking_active;
}

FString UConfigManager::GetHttpHealthUrl() const {
    return FString::Printf(TEXT("http://%s:%d%s"), *server.host, server.port, *server.http_health_path);
}

FString UConfigManager::GetWebSocketUrl() const {
    return FString::Printf(TEXT("ws://%s:%d%s"), *server.host, server.port, *server.websocket_path);
}

// --- INI Read Helpers ---
void UConfigManager::ReadFloat(const FString& section, const FString& key, float& out) const {
    GConfig->GetFloat(*section, *key, out, GetConfigFilePath());
}

void UConfigManager::ReadInt(const FString& section, const FString& key, int32& out) const {
    GConfig->GetInt(*section, *key, out, GetConfigFilePath());
}

void UConfigManager::ReadBool(const FString& section, const FString& key, bool& out) const {
    GConfig->GetBool(*section, *key, out, GetConfigFilePath());
}

void UConfigManager::ReadString(const FString& section, const FString& key, FString& out) const {
    GConfig->GetString(*section, *key, out, GetConfigFilePath());
}

// --- INI Write Helpers ---
void UConfigManager::WriteFloat(const FString& section, const FString& key, float val) {
    GConfig->SetFloat(*section, *key, val, GetConfigFilePath());
}

void UConfigManager::WriteInt(const FString& section, const FString& key, int32 val) {
    GConfig->SetInt(*section, *key, val, GetConfigFilePath());
}

void UConfigManager::WriteBool(const FString& section, const FString& key, bool val) {
    GConfig->SetBool(*section, *key, val, GetConfigFilePath());
}

void UConfigManager::WriteString(const FString& section, const FString& key, const FString& val) {
    GConfig->SetString(*section, *key, *val, GetConfigFilePath());
}