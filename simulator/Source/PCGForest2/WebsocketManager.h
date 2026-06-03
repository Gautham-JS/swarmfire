// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "IWebSocket.h"
#include "WebSocketsModule.h"
#include "Modules/ModuleManager.h"
#include "ConfigManager.h"

/**
 * 
 */
class PCGFOREST2_API WebsocketManager {
    private:
        static WebsocketManager* instance;
        TSharedPtr<IWebSocket> socket;

        // Private constructor for singleton
        WebsocketManager();
    public:
        ~WebsocketManager();

        static WebsocketManager* Get();
        static void Shutdown();
        static void Clean();
        void Connect(const FString& Uurl);
        void SendJsonMessage(const FString& type, const FString& message);
        void SendRawMessage(const TArray<uint8>& data);
        void SendObservation(UTextureRenderTarget2D* render_target, int32 step);

    private:
        // Callbacks
        void OnConnected();
        void OnConnectionError(const FString& err);
        void OnClosed(int32 rc, const FString& reason, bool bWasClean);
        void OnMessage(const FString& message);
        void HandleObsRequest(const TSharedPtr<FJsonObject>& JsonObject);
};
