
#include "TeleportCameraCommand.h"
#include "DuckiebotsSim.h"
#include "HolodeckGameMode.h"

void UTeleportCameraCommand::Execute() {
	UE_LOG(LogDuckiebotsSim, Log, TEXT("TeleportCameraCommand::Execute teleport camera command"));

	if (StringParams.size() != 0) {
		UE_LOG(LogDuckiebotsSim, Error, TEXT("Unexpected argument length found in TeleportCameraCommand. Command not executed."));
		return;
	}

	UWorld* World = Target->GetWorld();
	float UnitsPerMeter = World->GetWorldSettings()->WorldToMeters;
	FVector Location = FVector(NumberParams[0], NumberParams[1], NumberParams[2]);
	Location = ConvertLinearVector(Location, ClientToUE);
	FVector Rotation = FVector(NumberParams[3], NumberParams[4], NumberParams[5]);
	Rotation = ConvertAngularVector(Rotation, ClientToUE);


	AHolodeckGameMode* Game = static_cast<AHolodeckGameMode*>(Target);

	Game->TeleportCamera(Location, Rotation);
}
