from XAgentServer.envs import XAgentServerEnv


def email_content(user):
    """Generates the email content in HTML format to be sent to the user after signing up.

    This function creates a welcome message containing user's account information like their email,
    token and a link to activate their account.

    Args:
        user (dict): A dictionary containing user's information. The dictionary must contain the keys 
                     'name', 'email', 'token', and 'user_id'.

    Returns:
        str: A string representing the email content in HTML format.

    Examples:
        >>> user = {'name': 'John Doe', 'email': 'john.doe@example.com', 'token': 'XYZ123', 'user_id': '12345'}
        >>> print(email_content(user))
    """
    return f"""
<body style="font-family: Arial, sans-serif;background-color: #f5f5f5;margin: 0; padding: 0;">
    <div style="background-color: #ffffff;margin: 0 auto;padding: 20px;border-radius: 10px;box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);">
        <h1 style="font-size: 28px;margin-bottom: 20px;">Hello {user['name']},</h1>
        <p style="font-size: 16px;line-height: 1.5;color: #333333;text-indent:2em;">Welcome to XAgent, your personal assistant! Thanks for signing up for XAgent. There are some information about your account:</p>
        <p style="font-size: 16px;line-height: 1.5;color: #333333;text-indent:2em;">Your XAgent Account: <b>{user["email"]}</b></p>
        <p style="font-size: 16px;line-height: 1.5;color: #333333;text-indent:2em;">You need to use this token for authentication on subsequent logins: </p>
        <p style="font-size: 16px;line-height: 1.5;color: #333333;text-indent:2em;">You need to use this token for authentication on subsequent logins: </p>
        <p style="font-size: 16px;line-height: 1.5;color: #333333;text-indent:2em;">Token: <b>{user["token"]}</b></p>
        
        <p style="font-size: 16px;line-height: 1.5;color: #333333;text-indent:2em;">Next is an activation link. You need to click on this link to activate your account. After that, you will be able to use XAgent happily:<a href="{XAgentServerEnv.Email.auth_server}/auth?user_id={user["user_id"]}&token={user["token"]}">{XAgentServerEnv.Email.auth_server}/auth?user_id={user["user_id"]}&token={user["token"]}</a>! This Verification link will expire in 7 days.</p>
        
        <p>If you have any questions, please contact us at yourxagent@gmail.com .</p>
        <p style="margin-top: 20px;font-size: 14px;color: #999999;text-indent:2em;">Best wishes!</p>
        <p style="margin-top: 20px;font-size: 14px;color: #999999;">XAgent Team</p>
    </div>
</body>"""