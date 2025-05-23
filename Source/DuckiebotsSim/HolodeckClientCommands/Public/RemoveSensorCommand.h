#pragma once

#include "DuckiebotsSim.h"

#include "Command.h"
#include "RemoveSensorCommand.generated.h"

/**
* RemoveSensorCommand
* Command used to remove a sensor from an agent
* Use of this command should be infrequent.
*
* StringParameters expect two arguments, the agent name, and sensor name.
*/
UCLASS()
class DUCKIEBOTSSIM_API URemoveSensorCommand : public UCommand
{
	GENERATED_BODY()

public:
	//See UCommand for the documentation of this overridden function.
	void Execute() override;

};
