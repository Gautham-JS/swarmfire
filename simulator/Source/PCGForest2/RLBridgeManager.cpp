#include "RLBridgeManager.h"
#include "WebSocketsModule.h"
#include "Json.h"
#include "Misc/Base64.h"
#include "ImageUtils.h"
#include "Engine/TextureRenderTarget2D.h"
#include "Modules/ModuleManager.h"

URLBridgeManager* URLBridgeManager::instance = nullptr;

URLBridgeManager* URLBridgeManager::Get() {
    if (!instance || !instance->IsValidLowLevel()) {
        instance = NewObject<URLBridgeManager>();
        instance->AddToRoot();
    }
    return instance;
}

void URLBridgeManager::Connect(const FString& url) {
    if (!FModuleManager::Get().IsModuleLoaded(TEXT("WebSockets")))
        FModuleManager::Get().LoadModule(TEXT("WebSockets"));

    this->socket = FWebSocketsModule::Get().CreateWebSocket(url, TEXT("ws"));

    this->socket->OnConnected().AddUObject(this, &URLBridgeManager::HandleConnected);
    this->socket->OnConnectionError().AddUObject(this, &URLBridgeManager::HandleError);
    this->socket->OnClosed().AddUObject(this, &URLBridgeManager::HandleClosed);
    this->socket->OnMessage().AddUObject(this, &URLBridgeManager::HandleMessage);

    this->socket->Connect();

    UE_LOG(LogTemp, Log, TEXT("[RL] Connecting to %s"), *url);
}

void URLBridgeManager::Disconnect() {
    if (this->socket.IsValid())
        this->socket->Close();
}

void URLBridgeManager::SendObservation(
    UTextureRenderTarget2D* seg_rt,
    FVector2D norm_pos,
    FVector2D norm_vel,
    FVector2D vol_size_uu,
    float     elevation,
    float     fov_degrees,
    TArray<FVector2D> fire_norm_positions,
    bool      inside_pcg,
    int32     step_id,
    bool      episode_done) {
    if (!this->connected) return;

    TSharedPtr<FJsonObject> root = MakeShareable(new FJsonObject);
    root->SetStringField(TEXT("type"), TEXT("observation"));
    root->SetNumberField(TEXT("step_id"), step_id);
    root->SetBoolField(TEXT("done"), episode_done);

    // Segmentation mask 
    root->SetStringField(TEXT("seg_b64"), EncodeRT(seg_rt));
    root->SetNumberField(TEXT("seg_w"),
        seg_rt ? seg_rt->SizeX : 512);
    root->SetNumberField(TEXT("seg_h"),
        seg_rt ? seg_rt->SizeY : 512);

    // Drone state
    TSharedPtr<FJsonObject> state = MakeShareable(new FJsonObject);
    state->SetNumberField(TEXT("norm_x"), norm_pos.X);
    state->SetNumberField(TEXT("norm_y"), norm_pos.Y);
    state->SetNumberField(TEXT("norm_vx"), norm_vel.X);
    state->SetNumberField(TEXT("norm_vy"), norm_vel.Y);
    state->SetBoolField(TEXT("inside"), inside_pcg);
    root->SetObjectField(TEXT("state"), state);

    // Volume / camera params
    TSharedPtr<FJsonObject> params = MakeShareable(new FJsonObject);
    params->SetNumberField(TEXT("vol_size_x"), vol_size_uu.X);
    params->SetNumberField(TEXT("vol_size_y"), vol_size_uu.Y);
    params->SetNumberField(TEXT("elevation"), elevation);
    params->SetNumberField(TEXT("fov_deg"), fov_degrees);
    root->SetObjectField(TEXT("camera_params"), params);

    // Fire positions
    TArray<TSharedPtr<FJsonValue>> fire_arr;
    for (const FVector2D& fp : fire_norm_positions) {
        TSharedPtr<FJsonObject> fobj = MakeShareable(new FJsonObject);
        fobj->SetNumberField(TEXT("nx"), fp.X);
        fobj->SetNumberField(TEXT("ny"), fp.Y);
        fire_arr.Add(MakeShareable(
            new FJsonValueObject(fobj)));
    }
    root->SetArrayField(TEXT("fire_positions"), fire_arr);

    FString out;
    TSharedRef<TJsonWriter<>> w = TJsonWriterFactory<>::Create(&out);
    FJsonSerializer::Serialize(root.ToSharedRef(), w);
    this->socket->Send(out);
}

void URLBridgeManager::SendResetConfirm(FVector2D vol_size_uu, float elevation, float fov_degrees, FVector2D drone_start_norm) {
    if (!this->connected) return;

    TSharedPtr<FJsonObject> root = MakeShareable(new FJsonObject);
    root->SetStringField(TEXT("type"), TEXT("reset_confirm"));
    root->SetNumberField(TEXT("vol_size_x"), vol_size_uu.X);
    root->SetNumberField(TEXT("vol_size_y"), vol_size_uu.Y);
    root->SetNumberField(TEXT("elevation"), elevation);
    root->SetNumberField(TEXT("fov_deg"), fov_degrees);
    root->SetNumberField(TEXT("drone_start_nx"), drone_start_norm.X);
    root->SetNumberField(TEXT("drone_start_ny"), drone_start_norm.Y);

    FString out;
    TSharedRef<TJsonWriter<>> w = TJsonWriterFactory<>::Create(&out);
    FJsonSerializer::Serialize(root.ToSharedRef(), w);
    this->socket->Send(out);
}

void URLBridgeManager::SendEpisodeDone(int32 total_steps, float total_reward) {
    if (!this->connected) return;

    TSharedPtr<FJsonObject> root = MakeShareable(new FJsonObject);
    root->SetStringField(TEXT("type"), TEXT("episode_done"));
    root->SetNumberField(TEXT("total_steps"), total_steps);
    root->SetNumberField(TEXT("total_reward"), total_reward);

    FString out;
    TSharedRef<TJsonWriter<>> w = TJsonWriterFactory<>::Create(&out);
    FJsonSerializer::Serialize(root.ToSharedRef(), w);
    this->socket->Send(out);
}

FString URLBridgeManager::EncodeRT(UTextureRenderTarget2D* rt) {
    if (!rt) return TEXT("");

    FTextureRenderTargetResource* res = rt->GameThread_GetRenderTargetResource();
    if (!res) return TEXT("");

    TArray<FColor> pixels;
    if (!res->ReadPixels(pixels)) return TEXT("");

    for (FColor& p : pixels) p.A = 255;

    TArray<uint8, FDefaultAllocator64> png;
    FImageUtils::PNGCompressImageArray(rt->SizeX, rt->SizeY, pixels, png);

    // Convert to standard TArray<uint8> for Base64 encoding
    TArray<uint8> png_standard(png.GetData(), png.Num());
    return FBase64::Encode(png_standard);
}

void URLBridgeManager::HandleConnected() {
    this->connected = true;
    UE_LOG(LogTemp, Log, TEXT("[RL] Connected to Python RL server"));
    OnRLConnected.Broadcast();

    // Send handshake
    TSharedPtr<FJsonObject> hs = MakeShareable(new FJsonObject);
    hs->SetStringField(TEXT("type"), TEXT("ue5_ready"));
    hs->SetStringField(TEXT("client"), TEXT("UE5_RLBridge"));

    FString out;
    TSharedRef<TJsonWriter<>> w = TJsonWriterFactory<>::Create(&out);
    FJsonSerializer::Serialize(hs.ToSharedRef(), w);
    this->socket->Send(out);
}

void URLBridgeManager::HandleError(const FString& error) {
    this->connected = false;
    UE_LOG(LogTemp, Error, TEXT("[RL] WS Error: %s"), *error);
}

void URLBridgeManager::HandleClosed(int32 code, const FString& reason, bool clean) {
    this->connected = false;
    UE_LOG(LogTemp, Log, TEXT("[RL] WS Closed: %s"), *reason);
}

void URLBridgeManager::HandleMessage(const FString& message) {
    TSharedPtr<FJsonObject> json;
    TSharedRef<TJsonReader<>> r =
        TJsonReaderFactory<>::Create(message);

    if (!FJsonSerializer::Deserialize(r, json) || !json.IsValid()) return;

    FString type = json->GetStringField(TEXT("type"));

    if (type == TEXT("action")) ParseAction(json);
    else if (type == TEXT("reset")) ParseReset(json);
    else if (type == TEXT("ping")) {
        TSharedPtr<FJsonObject> pong = MakeShareable(new FJsonObject);
        pong->SetStringField(TEXT("type"), TEXT("pong"));
        FString out;
        TSharedRef<TJsonWriter<>> w = TJsonWriterFactory<>::Create(&out);
        FJsonSerializer::Serialize(pong.ToSharedRef(), w);
        this->socket->Send(out);
    }
}

void URLBridgeManager::ParseAction(TSharedPtr<FJsonObject> json) {
    FRLAction action;
    action.dx = (int32)json->GetNumberField(TEXT("dx"));
    action.dy = (int32)json->GetNumberField(TEXT("dy"));
    action.step_id = (int32)json->GetNumberField(TEXT("step_id"));

    this->pending_action = action;
    this->waiting_for_action = false;

    OnActionReceived.Broadcast(action);

    UE_LOG(LogTemp, Log,
        TEXT("[RL] Action: dx=%d dy=%d step=%d"),
        action.dx, action.dy, action.step_id);
}

void URLBridgeManager::ParseReset(TSharedPtr<FJsonObject> json) {
    UE_LOG(LogTemp, Log, TEXT("[RL] Reset requested by Python"));
    OnResetRequested.Broadcast();
}
