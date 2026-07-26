class TaskList:
    def __init__(self):
        self.tasks = []

    def add(self, title):
        self.tasks.append({"title": title, "done": False})

    def complete(self, index):
        self.tasks[index]["done"] = True

    def pending(self):
        return [task for task in self.tasks if not task["done"]]
