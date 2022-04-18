from audioop import reverse
import operator
from simulator.customEnums import Scheduling, Status
from simulator.task import Task
# from task import Task


# for simulator initialization


class Simulator:
    def __init__(self, taskset, processors, horizon, scheduling):
        self.eventList = []
        self.taskset = taskset
        self.processors = processors
        self.processorType = 2
        self.scheduling = scheduling
        self.stopSimulator = False
        self.currentTime = 0
        self.releasedTasks = []
        self.error = None

        self.initState(horizon)

        self.status = {
            "released": 0,
            "worked": 0,
            "idle": 0,
            "finished": 0,
            "total": 0
        }

        for t in taskset:
            self.status["total"] += len(t) * len(t[0])

    class eventClass(object):
        # This is the class of events
        def __init__(self, case, delta, idx):
            self.eventType = case
            self.delta = delta
            self.idx = idx

        def case(self):
            if self.eventType == 0:
                return "release"
            elif self.eventType == 1:
                return "deadline"
            elif self.eventType == 2:
                return "period release"
            elif self.eventType == 3:
                return "end"

        def updateDelta(self, elapsedTime):
            self.delta = self.delta - elapsedTime

    def nextScheduledTask(self):
        if self.scheduling == Scheduling.EDF.value:
            # edf
            return self.earliestDeadlineFirst()

    def earliestDeadlineFirst(self):
        self.releasedTasks.sort(key=operator.itemgetter("deadline"))
        self.releasedTasks.sort(key=operator.itemgetter("status"), reverse=True)
        scheduledTasks = self.releasedTasks[0:self.processors]
        return scheduledTasks

    def release(self, idx):
        tasks = self.getTasksToId([idx])
        self.doRelease(tasks[0])
        self.eventList = sorted(self.eventList, key=operator.attrgetter('delta'))

    def doRelease(self, task):
        task["task"] = Task(f'({task["id"][0]}, {task["id"][1]}, {task["id"][2]})', -1, task["deadline"], task["execution"])
        task["status"] = Status.RELEASED.value
        self.releasedTasks.append(task)
        
        if self.currentTime > task["deadline"]:
            print("")
            print(f'Task {task["id"]} missed his deadline! Was not even released!')
            self.error = {"id": task["id"], "released": False, "deadline": task["deadline"]}
            self.stopSimulator = True

        self.eventList.append(self.eventClass(1, task["deadline"], task["id"]))

    def deadline(self, idx):
        # check if the targeted task in the table has workload.
        tasks = self.getTasksToId([idx])
        self.checkDeadline(tasks[0])

    def checkDeadline(self, task):
        if not "task" in task.keys():
            print("")
            print(f'Task {task["id"]} missed his deadline! Was not even released!')
            self.error = {"id": task["id"], "released": False, "deadline": task["deadline"]}
            self.stopSimulator = True
        else:
            task = task["task"]

            if task.workload != 0 and task.deadline <= self.currentTime:
                print("")
                print(f'Task {task.id} missed his deadline! Remaining Workload: {task.workload}, Deadline: {task.deadline}')
                self.error = {"id": task["id"], "released": True, "remaining": task.workload, "deadline": task.deadline}
                self.stopSimulator = True


    def periodRelease(self, idxs):
        tasks = self.getTasksToId(idxs)
        for task in tasks:
            self.doRelease(task)

        self.eventList = sorted(self.eventList, key=operator.attrgetter('delta'))

    def end(self, _):
        if len(self.releasedTasks) > 0:
            print("")
            print(f'{len(self.releasedTasks)} Tasks missed their deadline!')
            self.error = {"notFinished": len(self.releasedTasks), "deadline": 100}
            self.stopSimulator = True

    def event_to_dispatch(self, event):
        # take out the delta from the event
        self.elapsedTime(event)

        # execute the corresponding event functions
        switcher = {
            0: self.release,
            1: self.deadline,
            2: self.periodRelease,
            3: self.end
        }
        func = switcher.get(event.eventType, lambda: "ERROR")
        # execute the event
        func(event.idx)

    def elapsedTime(self, event):
        delta = event.delta - self.currentTime
        self.runElapsedTime(delta)

    def runElapsedTime(self, delta):
        self.currentTime += delta
        while (delta):
            # get Tasks, that should be scheduled according to chosen algorithm
            tasks = self.nextScheduledTask()
            # calculate lowest workload time
            tmpDelta = self.findLowestJob(tasks, delta)
            for task in tasks:
                if not task == None:
                    # This lines stops preemptive. There can't be more than amount processors tasks
                    # because they are always chosen first
                    task["status"] = Status.STARTED.value
                    # update workload with lowest workload time
                    ended = task["task"].updateWorkload(tmpDelta)
                    self.status["worked"] += tmpDelta
                    # do extra stuff if a task ended
                    if ended:
                        #delta = 0 #stop because there could be new release
                        #self.checkDeadline(task)

                        self.status["finished"] += 1
                        try:
                            # remove task for performance reasons
                            self.releasedTasks.remove(task)
                        except ValueError:
                            print(f'Trying to remove a Task, but it wasnt there...')
                            print(f'{self.currentTime}')
                            print(f'{task}')
                            exit()
                        task["status"] = Status.STOPPED.value

                        # check all successors and decrease predecessor counter
                        for (t, s, p) in task["successor"]:
                            self.taskset[t][s][p]["predecessor"] -= 1

                            # if predecessor counter is zero, release
                            if self.taskset[t][s][p]["predecessor"] == 0:
                                periodSubTask = self.taskset[t][s][p]
                                # if allowed release is lower than current time, immediately release
                                if periodSubTask["release"] < self.currentTime:
                                    self.release(periodSubTask["id"])
                                # otherwise set earliest possible release as release
                                else:
                                    self.eventList.append(self.eventClass(0, periodSubTask["release"] - self.currentTime, periodSubTask["id"]))
                                    self.eventList = sorted(self.eventList, key=operator.attrgetter('delta'))
            
            # add idle for empty processors
            self.status["idle"] += tmpDelta * (self.processors - len(tasks))
            # lower delta by the time simulated
            delta -= tmpDelta

        if delta < 0:
            print('big fat error**************************')

    def findLowestJob(self, tasks, delta):
        minDelta = delta

        for task in tasks: 
            if (not task == None) and (minDelta > task["task"].workload):
                minDelta = task["task"].workload

        return minDelta

    def getTasksToId(self, ids):
        tasks = []
        for (t, s, p) in ids:
            if not t == -1:
                tasks.append(self.taskset[t][s][p])
            else:
                tasks.append(None)
        return tasks

    def getNextEvent(self):
        # get the next event from the event list
        if len(self.eventList) == 0:
            return None
        event = self.eventList.pop(0)
        return event

    def initState(self, horizon):
        # init
        self.eventList = []
        # check for releasable subtasks
        self.initEvents(horizon)

    def dispatcher(self, stopTime, set_id):
        # Stop when the time of maxPeriod * jobnum is overstepped or on miss.
        while(self.currentTime <= stopTime and not self.stopSimulator):
            if len(self.eventList) == 0:
                break
            else:
                print(f'{set_id}/{100}: Current Time: {self.currentTime:.8f}', end="\r")# end="\r"
                e = self.getNextEvent()
                self.event_to_dispatch(e)

        #check for remaining deadline with potential miss
        e = self.getNextEvent()

        if not e == None:
            while(e.delta >= self.currentTime and not self.stopSimulator):
                # only deadlines, releases dont create new deadlines, that can be missed
                if(e.case() == "deadline"):
                    self.event_to_dispatch(e)
                if(len(self.eventList) != 0):
                    e = self.getNextEvent()
                else:
                    break

        if not self.stopSimulator:
            print("")
        print(f'Has Deadline miss: {self.hasDeadlineMiss()}')
        show_stats = False
        if show_stats:
            print(f'Stats: Time {self.currentTime}, Worked {self.status["worked"]}, Idle {self.status["idle"]}, Util {self.status["worked"]/(self.currentTime * self.processors)}')
            print(f'Idle + Worked: {self.status["idle"] + self.status["worked"]}, Time * Processors: {self.currentTime * self.processors}')


    def hasDeadlineMiss(self):
        return self.stopSimulator

    # add all deadlines and the first releases
    def initEvents(self, horizon):
        periodReleases = [[] for _ in range(horizon)]
        # go through all first subtasks
        for (t, task) in enumerate(self.taskset):
            for (p, periodSubTask) in enumerate(task[0]):
                periodReleases[periodSubTask["release"]].append(periodSubTask["id"])

        for (p, list) in enumerate(periodReleases):
            self.eventList.append(self.eventClass(2, p, list))

        self.eventList.append(self.eventClass(3, horizon, None))
        self.eventList = sorted(self.eventList, key=operator.attrgetter('delta'))

    def jsonStats(self, status, objectives, time_result, incomplete_solver, set_id):
        result = {
            "id": set_id,
            "solution": status,
            "objectives": objectives,
            "times": time_result,
            "time": self.currentTime,
            "worked": self.status["worked"],
            "idle": self.status["idle"],
            "totalProcessorTime": self.status["idle"] + self.status["worked"],
            "totalPossibleTime": self.currentTime * self.processors,
            "util": self.status["worked"]/(self.currentTime * self.processors),
            "error": self.error,
            "usedLastSolver": incomplete_solver
        }

        return result


    def printEvents(self):
        event = self.eventList[0]
        for event in self.eventList:
                if event.eventType == 1:
                    print("Event: " + str(event.idx) + ", Type: " + str(event.eventType) + ", Delta: " + str(event.delta))