# locust -f locustfile.py --host=https://7iz1uma2gk.execute-api.us-east-1.amazonaws.com
# use this command line to run

from locust import HttpUser, between, task

class MyUser(HttpUser):
    wait_time = between(1, 2.5)
    
    @task
    def invoke_endpoint(self):
        data = {
            "input": [[1.0, 2.0, 3.0, 4.0]]
        }
        headers = {'Content-Type': 'application/json'}
        self.client.post("/test", json=data, headers=headers)
