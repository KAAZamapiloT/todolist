use std::io::{self, Write};

struct TodoList {
    tasks: Vec<String>,
}

impl TodoList {
    fn new() -> Self {
        TodoList { tasks: Vec::new() }
    }

    fn add_task(&mut self, task: String) {
        self.tasks.push(task);
    }

    fn list_tasks(&self) {
        if self.tasks.is_empty() {
            println!("No tasks in the list.");
        } else {
            for (index, task) in self.tasks.iter().enumerate() {
                println!("{}. {}", index + 1, task);
            }
        }
    }
}

fn main() {
    let mut todo_list = TodoList::new();

    loop {
        println!("\nTodo List App");
        println!("1. Add Task");
        println!("2. List Tasks");
        println!("3. Exit");
        print!("Choose an option: ");
        io::stdout().flush().unwrap();

        let mut choice = String::new();
        io::stdin().read_line(&mut choice).unwrap();

        match choice.trim() {
            "1" => {
                print!("Enter task: ");
                io::stdout().flush().unwrap();
                let mut task = String::new();
                io::stdin().read_line(&mut task).unwrap();
                todo_list.add_task(task.trim().to_string());
            }
            "2" => {
                todo_list.list_tasks();
            }
            "3" => break,
            _ => println!("Invalid option. Try again."),
        }
    }
}
