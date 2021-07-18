import os
import sys
from app.rotator import app

def run(env, port_num):
	# initialize Flask application
	# application localhost port
	port = int(os.environ.get("PORT", port_num))
	# run Flask application
	if env == 'production':
		# run in production mode
		app.run(debug=False, host='0.0.0.0', port=port)
	elif env == 'development':
		# run in developement mode
		app.run(debug=True, host='0.0.0.0', port=port)
	else:
		# invalid environment
		raise Exception('Invalid environment !')


if __name__ == "__main__":
	# initialize Flask application
	#run(sys.argv[1], sys.argv[2])
	run("production", 5001)
