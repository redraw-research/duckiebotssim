// Written by joshgreaves

#include "HolodeckWorldSettings.h"
#include "DuckiebotsSim.h"

float AHolodeckWorldSettings::FixupDeltaSeconds(float DeltaSeconds, float RealDeltaSeconds) {
	return ConstantTimeDeltaBetweenTicks;
}

float AHolodeckWorldSettings::GetConstantTimeDeltaBetweenTicks() {
	return ConstantTimeDeltaBetweenTicks;
}

void AHolodeckWorldSettings::SetConstantTimeDeltaBetweenTicks(float Delta) {
	ConstantTimeDeltaBetweenTicks = Delta;
}
