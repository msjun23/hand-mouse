import pyautogui

def moveCursor(height, width, x, y):
	monitor_width, monitor_height = pyautogui.size()

	move_to_x = x * (monitor_width / width)
	move_to_y = y * (monitor_height / height)

	if move_to_x <= 0:
		move_to_x = 10
	if move_to_x >= monitor_width:
		move_to_x = monitor_width - 10
	if move_to_y <= 0:
		move_to_y = 10
	if move_to_y >= monitor_height:
		move_to_y = monitor_height - 10

	try:
		#print(int(move_to_x), int(move_to_y))
		#pyautogui.moveTo(move_to_x, move_to_y)
		pyautogui.moveTo(int(move_to_x), int(move_to_y))
	except Exception as e:
		pyautogui.moveTo(monitor_width/2, monitor_height/2)

def executeCommand(command):
	print(command)

	if command == 'Cursor':
		pass
	elif command == 'Enter':
		pyautogui.press('enter')
	elif command == 'Esc':
		pyautogui.press('esc')
	elif command == 'LeftClick':
		pyautogui.click()
	elif command == 'WheelClick':
		pyautogui.click(button='middle')
	elif command == 'RightClick':
		pyautogui.click(button='right')
	elif command == 'ScrollUp':
		pyautogui.scroll(1)
	elif command == 'ScrollDown':
		pyautogui.scroll(-1)
	elif command == 'Spacebar':
		pyautogui.press('space')