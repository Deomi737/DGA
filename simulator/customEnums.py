from enum import Enum

#   usage:
#   to get the name, type .name
#   e.g. print(ProcessorType.GLOBAL.name)
#   ->  GLOBAL
#   to get the value, type .value
#   e.g. print(ProcessorType.GLOBAL.value)
#   ->  2

class ProcessorType(Enum):
    #  0        1           2
    SINGLE, PARTITIONED, GLOBAL = range(3)



class Scheduling(Enum):
    #   0      1    2   3   4
    STANDARD, FTP, EDF, DM, RM = range(5)


class Status(Enum):
    #   0          1          2      3   
    UNRELEASED, RELEASED, STARTED, STOPPED = range(4)

def reverseMapProcessorType(processorType):
    #print(processorType)

    if processorType == ProcessorType.SINGLE.value:
        return ProcessorType.SINGLE.name
    elif processorType == ProcessorType.PARTITIONED.value:
        return ProcessorType.PARTITIONED.name
    elif processorType == ProcessorType.GLOBAL.value:
        return ProcessorType.GLOBAL.name
    else:
        raise ValueError()

def reverseMapScheduling(scheduling):
    if scheduling == Scheduling.STANDARD.value:
        return Scheduling.STANDARD.name
    elif scheduling == Scheduling.FTP.value:
        return Scheduling.FTP.name
    elif scheduling == Scheduling.EDF.value:
        return Scheduling.EDF.name
    elif scheduling == Scheduling.DM.value:
        return Scheduling.DM.name
    elif scheduling == Scheduling.RM.value:
        return Scheduling.RM.name
    else:
        raise ValueError()

def reverseMapStatus(status):
    if status == Status.UNRELEASED.value:
        return Status.UNRELEASED.name
    elif status == Status.RELEASED.value:
        return Status.RELEASED.name
    elif status == Status.STARTED.value:
        return Status.STARTED.name
    elif status == Status.STOPPED.value:
        return Status.STOPPED.name
    else:
        raise ValueError()
