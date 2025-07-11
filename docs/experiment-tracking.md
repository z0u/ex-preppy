Some experiments are tracked using Aim. For remote experiments (in Modal), you'll need to start Aim before running them.

```
./go track
```

## Setting up authentication

The experiment tracker is configured to use Google as an authentication provider. You'll need to do some manual set-up the first time you use it.

1. Enable OAuth in your Google account

   1. Go to [Google Cloud Console](https://console.cloud.google.com) (available to anyone with a GMail account)

   2. Create a new project (e.g. "Modal Auth").

   3. In the hamburger menu: _APIs & Services_ → _OAuth consent screen_

   Click _Get started_. Continue through the creation wizard. Add youself as a test user.

   4. Then go to: _APIs & Services_ → _Credentials_ → _Create Credentials_ → _OAuth Client ID_

   - App type: _Web Application_

   - Add JS origins like:

     `https://<workspace>--track.modal.run`

     `https://<workspace>--track-dev.modal.run`

     > [!TIP]
     > Your workspace name is whatever appears in the URL of your Modal dashboard, e.g. `https://modal.com/apps/:workspace-name`

   - Add redirect URIs like:

     `https://<workspace>--track.modal.run/auth`

     `https://<workspace>--track-dev.modal.run/auth`

   Press _Create_, and **save the _Client secret_** before closing the dialog.

2. Add the auth details to Modal

   1. Go to [modal.com/secrets](https://modal.com/secrets)
   2. Create a new, custom secret. Call it `google-oauth`

      Keys:

      - `GOOGLE_CLIENT_ID`: see step 1.
      - `GOOGLE_CLIENT_SECRET`: see step 1.
      - `ALLOWED_EMAIL`: your email address, or a comma-separated list of email addresses
