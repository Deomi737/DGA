class Task:

    def __init__(self, taskid, period, deadline, execution):
        self.id = taskid
        self.period = period
        self.deadline = deadline
        self.execution = execution
        self.workload = execution

    def __getitem__(self,key):
        return getattr(self,key)

    def updateWorkload(self, delta) -> bool:
        if self.workload < delta:
            print(f'BUG: Actual workload of Task {self.id} is less than {delta}, current workload {self.workload}')
        
        if delta == self.workload:
            self.workload = 0
            return True
        else:
            self.workload -= delta
            return False

    def convertToArr(self):
        return [self.id, self.period, self.deadline, self.execution, self.workload]

    def __str__(self):
        return "id: " + str(self.id) + ", period: " + str(self.period) + ", deadline: " + str(self.deadline) + ", execution: " + str(self.execution) + ", workload: " + str(self.workload)
        
    def __repr__(self):
        return "id: " + str(self.id) + ", period: " + str(self.period) + ", deadline: " + str(self.deadline) + ", execution: " + str(self.execution) + ", workload: " + str(self.workload)