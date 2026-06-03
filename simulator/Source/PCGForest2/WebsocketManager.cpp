// Fill out your copyright notice in the Description page of Project Settings.


#include "WebsocketManager.h"
#include "WebSocketsModule.h"
#include "Json.h"
#include "JsonUtilities.h"
#include "ConfigManager.h"

#include "Components/SceneComponent.h"
#include "Engine/TextureRenderTarget2D.h"


WebsocketManager* WebsocketManager::instance = nullptr;

WebsocketManager::WebsocketManager() {}
WebsocketManager::~WebsocketManager() {}

WebsocketManager* WebsocketManager::Get() {
	if (!instance) {
		if (GEngine) {
			GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green,
				TEXT("[WS] SETTING UP SINGLETON")
			);
		}
		instance = new WebsocketManager();
	}
	else {
		UE_LOG(LogTemp, Warning, TEXT("[WS] [GET] VALID instance ptr exists"));
	}
	return instance;
}


void WebsocketManager::Shutdown() {
	if (instance) {
		delete instance; instance = nullptr;
	}
}

void WebsocketManager::Clean() {
	if (instance) {
		instance->Shutdown();
	}
}

void WebsocketManager::Connect(const FString &url) {
	FWebSocketsModule* module = &FModuleManager::LoadModuleChecked<FWebSocketsModule>(TEXT("WebSockets"));
	
	this->socket = module->CreateWebSocket(url);
	this->socket->OnConnected().AddRaw(this, &WebsocketManager::OnConnected);
	this->socket->OnConnectionError().AddRaw(this, &WebsocketManager::OnConnectionError);
	this->socket->OnClosed().AddRaw(this, &WebsocketManager::OnClosed);
	this->socket->OnMessage().AddRaw(this, &WebsocketManager::OnMessage);

	UE_LOG(LogTemp, Warning, TEXT("[WS] [CONNECT]: url - %s"), *url);
	this->socket->Connect();
}

void WebsocketManager::OnConnected() {
	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Green,
			TEXT("[WS] CONNECTION ESTABLISHED")
		);
	}
	UE_LOG(LogTemp, Warning, TEXT("[WS] CONNECTION ESTABLISHED"));
}

void WebsocketManager::OnMessage(const FString& message) {
	UE_LOG(LogTemp, Warning, TEXT("[WS] Received: %s"), *message);
	TSharedPtr<FJsonObject> JsonObject;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(message);

	if (FJsonSerializer::Deserialize(Reader, JsonObject) && JsonObject.IsValid()) {
		FString Value = JsonObject->GetStringField("message");
		UE_LOG(LogTemp, Warning, TEXT("[WS] Parsed message: %s"), *Value);
	}
	else {
		UE_LOG(LogTemp, Error, TEXT("[WS] Error parsing json message."));
	}
}

void WebsocketManager::OnConnectionError(const FString& err) {
	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Red,
			FString::Printf(TEXT("[WS] CONNECTION ERROR : %s"), *err)
		);
	}
	UE_LOG(LogTemp, Error, TEXT("[WS] CONNECTION ERROR : %s"), *err);
}

void WebsocketManager::OnClosed(int32 rc, const FString& reason, bool bWasClean) {
	if (GEngine) {
		GEngine->AddOnScreenDebugMessage(-1, 5.f, FColor::Orange,
			FString::Printf(TEXT("[WS] CONNECTION CLOSED : %s"), *reason)
		);
	}
	UE_LOG(LogTemp, Warning, TEXT("[WS] Connection Closed: %s"), *reason);
}


void WebsocketManager::SendJsonMessage(const FString& type, const FString& message) {
	if (!this->socket.IsValid()) {
		UE_LOG(LogTemp, Warning, TEXT("[WS] Socket not valid"));
		return;
	}
	TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
	JsonObject->SetStringField("type", type);
	JsonObject->SetStringField("message", message);

	FString OutputString;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
	FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);

	this->socket->Send(OutputString);
	UE_LOG(LogTemp, Warning, TEXT("[WS] Sent: %s"), *OutputString);
}

void WebsocketManager::SendRawMessage(const TArray<uint8>& data) {
	if (!this->socket.IsValid()) {
		UE_LOG(LogTemp, Warning, TEXT("[WS] Socket not valid"));
		return;
	}
	if (data.Num() == 0) {
		UE_LOG(LogTemp, Warning, TEXT("[WS] Empty data, not sending"));
		return;
	}
	this->socket->Send(data.GetData(), data.Num(), true); // true = binary
	UE_LOG(LogTemp, Warning, TEXT("[WS] Sent raw binary: %d bytes"), data.Num());
}